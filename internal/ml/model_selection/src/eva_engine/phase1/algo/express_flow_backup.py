from src.eva_engine.phase1.algo.alg_base import Evaluator
from src.eva_engine.phase1.utils.autograd_hacks import *
from torch import nn
from src.common.constant import Config
from functools import partial


class IntegratedHook:
    def __init__(self):
        self.originals = []
        self.perturbations = []
        self.Vs = []
        self.activation_map = {}
        self.is_perturbed = False

    def forward_hook(self, module, input, output):
        # Store the output based on whether it's perturbed or not
        if isinstance(module, nn.ReLU):
            if self.is_perturbed:
                self.perturbations.append(output)
            else:
                self.originals.append(output)

        # Save this output in the map using the module's ID
        self.activation_map[id(module)] = output

        # Register backward hook for gradient computation
        # todo: this will messed up the reference, result in the memory leak.
        # output.register_hook(lambda grad: self.backward_hook(grad, module))
        output.register_hook(partial(self.backward_hook, module=module))

    def backward_hook(self, grad, module):
        dz = grad  # gradient
        # Get the correct activation from the map
        activation = self.activation_map[id(module)]
        V = activation * abs(dz)  # product
        self.Vs.append(V)

    def calculate_trajectory_length(self, epsilon):
        # assert len(self.originals) == len(self.perturbations)
        trajectory_lengths = [abs(x_perturbed - x).norm() / epsilon for x, x_perturbed in
                              zip(self.originals, self.perturbations)]
        return trajectory_lengths

    def clear_all(self):
        self.originals.clear()
        self.perturbations.clear()
        self.Vs.clear()
        self.activation_map.clear()


class ExpressFlowEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def linearize(self, arch):
        signs = {}
        for name, param in arch.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(self, arch, signs):
        for name, param in arch.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor,
                 space_name: str) -> float:

        # Step 1: Linearize
        if space_name == Config.MLPSP:
            signs = self.linearize(arch.mlp)
            arch.mlp.double()
        else:
            signs = self.linearize(arch)
            arch.double()

        hook_obj = IntegratedHook()
        hooks = []
        for module in arch.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_obj.forward_hook))

        epsilon = 1e-5
        delta_x = torch.randn_like(batch_data) * epsilon

        # Forward pass with original input
        hook_obj.is_perturbed = False
        if space_name == Config.MLPSP:
            out = arch.forward_wo_embedding(batch_data.double())
        else:
            out = arch.forward(batch_data.double())

        # Forward pass with perturbed input
        hook_obj.is_perturbed = True
        if space_name == Config.MLPSP:
            _ = arch.forward_wo_embedding(batch_data.double() + delta_x)
        else:
            _ = arch.forward(batch_data.double() + delta_x)

        trajectory_lengths = hook_obj.calculate_trajectory_length(epsilon)

        # directly sum
        torch.sum(out).backward()

        # total_sum = self.compute_score(trajectory_lengths, hook_obj.Vs)
        total_sum = self.weighted_score(trajectory_lengths, hook_obj.Vs)
        # total_sum = self.weighted_score_traj(trajectory_lengths, hook_obj.Vs)
        # total_sum = self.weighted_score_width(trajectory_lengths, hook_obj.Vs)

        # Step 2: Nonlinearize
        if space_name == Config.MLPSP:
            self.nonlinearize(arch.mlp, signs)
        else:
            self.nonlinearize(arch, signs)

        # Remove the hooks
        for hook in hooks:
            hook.remove()
        del hooks
        hook_obj.clear_all()

        return total_sum

    def weighted_score(self, trajectory_lengths, Vs):
        trajectory_lengths.reverse()
        # Modify trajectory_lengths to ensure that deeper layers have smaller weights
        # For example, by taking the inverse of each computed trajectory length.
        inverse_trajectory_lengths = [1.0 / (length + 1e-6) for length in trajectory_lengths]

        # Normalize trajectory lengths if needed (this ensures the weights aren't too large)
        normalized_lengths = [length / sum(inverse_trajectory_lengths) for length in inverse_trajectory_lengths]

        # Use the normalized trajectory lengths as weights for your total_sum
        total_sum = sum(
            normalized_length * V.flatten().sum() * V.shape[1]
            for normalized_length, V in zip(normalized_lengths, Vs))
        total_sum = total_sum

        return total_sum

    def weighted_score_traj(self, trajectory_lengths, Vs):
        trajectory_lengths.reverse()
        # Modify trajectory_lengths to ensure that deeper layers have smaller weights
        # For example, by taking the inverse of each computed trajectory length.
        inverse_trajectory_lengths = [1.0 / (length + 1e-6) for length in trajectory_lengths]

        # Normalize trajectory lengths if needed (this ensures the weights aren't too large)
        normalized_lengths = [length / sum(inverse_trajectory_lengths) for length in inverse_trajectory_lengths]

        # Use the normalized trajectory lengths as weights for your total_sum
        total_sum = sum(
            normalized_length * V.flatten().sum() for normalized_length, V in zip(normalized_lengths, Vs))
        total_sum = total_sum

        return total_sum

    def weighted_score_width(self, trajectory_lengths, Vs):
        # Vs is a list of tensors, where each tensor corresponds to the product
        # V=z×∣dz∣ (where z is the activation and dz is the gradient) for every ReLU layer in model.
        # Each tensor in Vs has the shape (batch_size, number_of_neurons)
        # 1. aggregates the importance of all neurons in that specific ReLU module.
        # 2. only use the first half layers.

        # Sum over the second half of the modules,
        # Vs[i].shape[1]: number of neuron in the layer i
        total_sum = sum(V.flatten().sum() * V.shape[1] for V in Vs) / 10
        total_sum = total_sum
        return total_sum

    def compute_score(self, trajectory_lengths, Vs):
        # Vs is a list of tensors, where each tensor corresponds to the product
        # V=z×∣dz∣ (where z is the activation and dz is the gradient) for every ReLU layer in model.
        # Each tensor in Vs has the shape (batch_size, number_of_neurons)
        # 1. aggregates the importance of all neurons in that specific ReLU module.
        # 2. only use the first half layers.

        # Sum over the second half of the modules,
        # Vs[i].shape[1]: number of neuron in the layer i
        total_sum = sum(V.flatten().sum() for V in Vs) / 10
        total_sum = total_sum
        return total_sum
