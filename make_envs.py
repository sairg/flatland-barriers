import random

from flatland.envs.base_grid_env import BaseGridEnv
from flatland.envs import EnvUtils
from flatland.envs.barriers_env import BarriersEnv



### global (needs to be moved to a class)
np_random = None

# food locations (at a corner)
food_locs = None
num_locs = None

##

# make environments, lists of lists. The outer is an initial
# environment (where barriers and possibly goal/food is
# different). Each inner list is a sequence of 2D grids, corresponding
# to days, where barriers and possibly food could change.
def make_environments(args):
    global food_locs, np_random
    
    human_mode = None  # render mode
    if args.do_human:
        human_mode = 'human' # enable human mode for viewing
    envs = []
    seed = args.seed

    np_random = random.Random()
    np_random.seed(seed)

    for i in range(args.outer):

        if 1: # reset possible food locations?
            food_locs = None
            
        days = [] #
        if i > 0 and seed is not None:
            seed += 1

        size = args.size
        food_loc = get_food_loc(
            size, day=1, args=args)
        
        env = BarriersEnv(
            seed=seed,
            render_mode=human_mode,
            size=size,
            prop=args.prop,
            food_loc=food_loc,
            change_rate=args.chr,
            #np_random=np_random, rely on pygame's random
            place_at_random=False,
            multiple_foods=False,
            motion_noise=args.mnoise)
        
        env = EnvUtils.get_env_with_a_path(env)

        if env is None:
            continue
            
        j, num_days = 0, args.days
        while j < num_days:
            days.append(env) # first day
            if j >= num_days - 1:
                break
            assert not env.multiple_foods
            env = env.make_copy() # deep copy
            env.day = env.get_day() + 1  # set day to day 1 ?
            env.change_env_map(change_type=1) # change map here
            if args.food_locs > 1:
                change_food_location(env, day=j+2, args=args)

            #Is there a path? try creating a new one till you get a path.
            env = EnvUtils.get_env_with_a_path(env)
            env.render_mode = human_mode # explicitly set it..
            if 0:
                print('\n#After making sure of path on day %d\n%s' % (
                    env.get_day(), env.barriers_to_text()))
             
            if env is None:  # no path was found? exit.
                break
                
            j += 1
        
        envs.append(days)

    return envs


# Change the location of food.
def change_food_location(env, day=None, args=None):
    if day is None:
        day = env.day
    env.remove_foods()
    x, y = get_food_loc(
        env.get_dimension(),  day=day, args=args)
    #x = env.get_dimension()-1
    env.place_food(False, x, y)


food_loc_idx = 0
# food_loc_idx = 2

def get_food_loc(
        size,  change_prob=0.0, day=None, args=None):
    global food_loc_idx, food_locs, num_locs, np_random

    # this assumes size won't change..
    if food_locs is None:
        food_locs = get_possible_food_locs(size, args)
        
    
    # round-robin, over the possible food locations    
    locs = food_locs # [ (x, x), (x, 0), (0, x), (0, 0)    ]
    locs = locs[:num_locs]
    #print('\n    num locs: ', num_locs, locs)

    if day is None:
        day = env.day
    if args.do_rr: # round-robin, over the possible food locations
        idx = day % num_locs
        return locs[food_loc_idx + idx]
    else: # just shuffle and return
        my_locs  = [x for x in locs] # copy
        np_random.shuffle(my_locs)
        #print('\n    my locs: ', my_locs)
        
        return my_locs[0]

# Returns possible food locations.
def get_possible_food_locs(size, args):
    global num_locs, np_random
    num_locs = 2 # number of food locations (on different days)
    if args is not None:
        num_locs = max(args.food_locs, 1) # at least 1.

    x = size - 1

    # corners only?
    if not args.do_interior:
        # corners only
        food_locs = [ (x, x), (x, 0), (0, x), (0, 0) ]
        if num_locs > 1:
            # if 1, always pick lower right? (don't shuffle?)
            np_random.shuffle( food_locs )
        num_locs = min(num_locs, len(food_locs))
        
    else: # pick the food goals at random
        # NOTE, todo: we shouldnt put food on home location..
        food_locs = set()
        for i in range(num_locs): # up to num_locs
            r = int( np_random.uniform(0, x) )
            c = int( np_random.uniform(0, x) )
            food_locs.add( (r, c) )
        food_locs = list(food_locs)
        num_locs = len(food_locs)
        
    return food_locs
        
    

    
###############

