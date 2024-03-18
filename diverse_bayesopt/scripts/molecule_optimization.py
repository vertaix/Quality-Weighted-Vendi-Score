import sys
sys.path.append("../")
import fire
from scripts.optimize import Optimize
from robot.tasks.molecules.molecule_objective import MoleculeObjective
from robot.tasks.molecules.load_data import load_molecule_train_data, compute_train_zs


class MoleculeOptimization(Optimize):
    """
    Run LOL-ROBOT Optimization for any Molecular Optimization Task using the SELFIES VAE 
    (Setup for any GuacaMol Scoring Benchmark Task or the Penalized LogP task)

    Args:
        path_to_vae_statedict: Path to state dict of pretrained SELFIES VAE,
        max_string_length: Limit on string length that can be generated by VAE (without a limit we can run into OOM issues)
    """
    def __init__(
        self,
        path_to_vae_statedict: str="../robot/tasks/molecules/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        max_string_length: int=400,
        **kwargs
    ):
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length

        super().__init__(**kwargs)

        # add args to method args dict to be logged by wandb
        self.method_args['molopt'] = locals()
        del self.method_args['molopt']['self']

    def initialize_objective(self):
        # initialize molecule objective
        self.objective = MoleculeObjective(
            task_id=self.task_id,
            path_to_vae_statedict=self.path_to_vae_statedict,
            max_string_length=self.max_string_length,
            smiles_to_selfies={}
        )

        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (a tensor of corresponding latent space points)
        '''
        assert self.num_initialization_points <= 20_000 
        smiles, selfies, zs, ys = load_molecule_train_data(
            task_id=self.task_id,
            num_initialization_points=self.num_initialization_points,
            path_to_vae_statedict=self.path_to_vae_statedict
        )
        self.init_train_z, self.init_train_y = zs, ys
        self.init_train_x = smiles
        if self.verbose:
            print("Loaded initial training data")
            print("train y shape:", self.init_train_y.shape)
            print(f"train x list length: {len(self.init_train_x)}\n")
        # create initial smiles to selfies dict
        init_smiles_to_selfies = {}
        for ix, smile in enumerate(self.init_train_x):
            init_smiles_to_selfies[smile] = selfies[ix]

        self.objective.smiles_to_selfies = init_smiles_to_selfies

        # if train zs have not been pre-computed for particular vae, compute them 
        #   by passing initialization selfies through vae 
        if self.init_train_z is None:
            self.init_train_z = compute_train_zs(
                self.objective,
                self.init_train_x,
            )

        return self 


if __name__ == "__main__":
    fire.Fire(MoleculeOptimization)
