from models import ActorCritic
from models import AddTemporalConstant
from models import DichoActorCritic

def build_model(obs_dim, action_dim, args):
        model_class = 'DichoActorCritic' if ('dicho' in vars(args) and
                                             args.dicho) else 'ActorCritic'

        model = eval(model_class)(obs_dim, action_dim)

        if args.remove_constant:
            model = AddTemporalConstant(model)

        return model
