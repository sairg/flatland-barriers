import numpy as np

# from PIL import Image

import pygame
import gymnasium as gym
from gymnasium import spaces
import time

"""       BaseGridEnv

Base-level environment for 2d grid world with food and barriers.
- Observations are an nxn window of grid centered at agent location
- Actions are left, right, up or down, when not blocked by barriers 
- Rewards are 1 for landing on a food cell and 0 otherwise
- Trip terminates when a reward is 1
- Moves are optionally in a relative or absolute orientation frame*
- Action is optionally displayed using pygame

To test, run python grid_env_test.py

Much of this can be reconfigured using class inheritance
- food and barriers can be distributed using a new create_env_map
- env map has 2 planes (food, barriers), can add/change planes
- observations, actions and rewards can all be changed
- food disappears when consumed, this can be changed
- pygame display can be changed (e.g. colors)

* Moves and orientation frames

> In absolute_orientation, a move is a step in an absolute direction:
right | up | left | down means step = (1, 0) | (0, 1) | (-1, 0) | (0, -1)
in the environment's global coordinate frame.
(so right or east means (x,y)->(x+1,y) , up means (x,y)->(x,y+1), etc. )

> In relative orientation (not used in the 2025 'When Remembering..'
paper, and not tested much), a move is a turn (rotation) of the
heading direction followed by a step forward in this direction: left |
forward | right | backward : rotate 90d left | 0d | 90d right | 180d
and then step in the direction of heading

Note: when in relative orientation, pygame displays agent as a triangle
pointing in the heading direction, otherwise as a circle
"""


class BaseGridEnv(gym.Env):
    def __init__(
        self,
        render_mode=None,
        size=32,
        orientation_frame='absolute_orientation',
        place_at_random=0,
        food_loc=None,
        time_cost=0,
        remove_consumed_food=True,
        motion_noise=0.0,  # useful when doing path integration.
        seed=None
    ):
        self.seed=seed
        if seed is not None:
            #print('HERE, seed was not None!', seed)
            super().reset(seed=seed)
            
        # For time sleeping..
        self.sleeper = 10        
        self.day = 1  # set day to day 1
        self.setup_environment_map(size, food_loc)
        self.food_loc = food_loc
        
        self.setup_observations()
        self.setup_actions(orientation_frame)
        self.setup_pygame(render_mode)
        self.step_count = 0
        self.time_cost = time_cost
        self.remove_consumed_food = remove_consumed_food
        # flag for placing food again
        self.place_at_random = place_at_random

        self.motion_noise_prob = motion_noise
        # If set to true, when injecting noise into motion, motion
        # will be biased (towards staying put).
        self.no_2steps_fwd = False
        # remain (don't change location).
        self.remain_if_barrier = 1
        # Probability of adding noise only in the direction of the
        # selected action (stay the place, or hope two steps in that
        # direction).
        self.noise_in_direction = 0.5

        #print('# BB place at rand (in base):', self.place_at_random)

    def get_day(self):
        # print('# get day is called', self.day)
        return self.day

    # or get_size()
    def get_dimension(self):
        return self.size
    
    def get_size(self):
        return self.size

    # Get count of an object type (food, barrier).
    def get_count(self, do_barrier=True):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if do_barrier:
                    has = self.env_map[self.barrier, i, j] > 0
                else:
                    has = self.env_map[self.food, i, j] > 0
                count += has
        return count
    
    def get_food_count(self):
        return self.get_count(do_barrier=False)
    
    def get_barrier_count(self):
        return self.get_count(do_barrier=True)

    def increment_day(self):
        # print('increment is called', self.day)
        self.day += 1

    def set_seed(self, seed):
        super().reset(seed=seed)
        
    # NOTE (Omid): seed appears not to have an effect? (is not used
    # currently?)
    def reset(self, seed=None, options=None):
        """reset agent's state, return first observation"""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.array(self.home)
        self._heading_direction = np.array(self.initial_heading)

        self.step_count = 0
        observation = self._get_obs()
        info = self._get_info()

        # Try placing food
        self.place_food(self.place_at_random)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    ##

    
    def step(self, action):
        """execute agent's action in the environment,
        return observation, reward and termination status"""
        self.update_location(action)

        # An episode is done if the agent has reached food
        terminated = (
            0
            < self.env_map[self.food, self._agent_location[0], self._agent_location[1]]
        )
        reward = 1 if terminated else -self.time_cost  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        if self.remove_consumed_food and (
            0
            < self.env_map[self.food, self._agent_location[0], self._agent_location[1]]
        ):
            self.env_map[
                self.food, self._agent_location[0], self._agent_location[1]
            ] = 0
            if self.render_mode is not None:
                env_map_image = self.create_env_map_image()
                self.env_map_surf = pygame.image.frombytes(
                    env_map_image.tobytes(), self.pygame_size, 'RGB'
                )

        # ? need to add time limit
        self.step_count = self.step_count + 1
        return observation, reward, terminated, False, info

    # ---- ENVIRONMENT MAP FUNCTIONS ----

    def setup_environment_map(self, size=32, food_loc=None):
        """define environment parameters, create environment"""
        self.size = size
        self.map_planes = (
            2  # number of planes (channels), currently 2: food and barriers
        )
        self.food, self.barrier = 0, 1  # semantics of the env map planes
        self.home = np.array(
            [self.size / 2, self.size / 2], dtype=np.int32
        )  # start position
        self.initial_heading = np.array([1, 0])  # initial heading direction

        # target_location is for optional observation
        # it is an array of target (food) locations
        self.env_map, self._target_location = self.create_env_map(
            food_loc=food_loc)
        #print("   HERE99, target loc:", self._target_location)
        self.step_count = 0
        
    ###
        
    ####
    
    def create_env_map(self):
        """create environment map with food and barrier distribution"""
        env_map_np = np.zeros((self.map_planes, self.size, self.size), dtype=np.uint8)
        # food_num uniformly randomly placed food points
        print('\n# **** HERE in super env map!!!\n\n')
        food_num = 30
        for i in range(food_num):
            x, y = self.np_random.integers(0, self.size - 1, size=2, dtype=int)
            if not np.array_equal(self.home, [x, y]):
                env_map_np[self.food, x, y] = 1
        length = 6
        barrier_num = 6
        # barrier_num/2 count of randomly placed vertical barrier segments of size length
        for i in range(int(barrier_num / 2)):
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size - length)
            if not inside(self.home, x, y, x, y + length):
                env_map_np[self.food, x, y : y + length] = 0
                env_map_np[self.barrier, x, y : y + length] = 1
        # barrier_num/2 count ofrandomly placed horizontal barrier segments of size length
        for i in range(int(barrier_num / 2)):
            y = np.random.randint(0, self.size)
            x = np.random.randint(0, self.size - length)
            if not inside(self.home, x, y, x + length, y):
                env_map_np[self.food, x : x + length, y] = 0
                env_map_np[self.barrier, x : x + length, y] = 1

        # pick the first food point for target location (targets are
        # optionally used for rewards)
        target_location = np.array(
            list(
                np.unravel_index(
                    np.argmax(env_map_np[self.food, :, :]),
                    env_map_np[self.food, :, :].shape,
                )
            )
        )
        return env_map_np, target_location

    def change_env_map(self, change=1):
        """stub for function that changes environment during experiments"""
        pass

    # ---- OBSERVATION FUNCTIONS ----

    def setup_observations(self):
        """define observation parameters and space"""
        self.obs_r = 1  # observation radius r means 2r+1 by 2r+1 agent-centered view
        self.view_width = self.obs_r * 2 + 1

        # Observations are dictionaries with the agent's and the
        # target's location and a 3x3 view of the map.  Each location
        # is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).  view of map is a
        # multibinary(2,3,3)
        self.observation_space = spaces.Dict(
            {
                'view': spaces.MultiBinary(
                    [self.map_planes, self.view_width, self.view_width], seed=42
                ),
                'agent': spaces.Box(
                    self.obs_r, self.size - self.obs_r - 1, shape=(2,), dtype=int
                ),
                'target': spaces.Box(
                    self.obs_r, self.size - self.obs_r - 1, shape=(2,), dtype=int
                ),
            }
        )

    def _get_obs(self):
        """sample an rxr window of the environment map, watch out for map edges,
        add agent and target location for optional use"""

        x, y = np.clip(self._agent_location, 0, self.size - 1)
        r = self.obs_r
        x1, y1 = np.clip(np.array([x - r, y - r]), 0, self.size - 1)
        x2, y2 = np.clip(np.array([x + r, y + r]), 0, self.size - 1)
        view = np.zeros((2, 2 * r + 1, 2 * r + 1))
        view[1, :, :] = 1
        view[:, x1 - x + r : x2 - x + r + 1, y1 - y + r : y2 - y + r + 1] = (
            self.env_map[:, x1 : x2 + 1, y1 : y2 + 1]
        )
        return {
            # some of these could be removed for experiments (eg to
            # focus the agent on informative features only).
            'view': view,
            'agent': np.array(self._agent_location),
            'target': self._target_location,
        }

    def _get_info(self):
        """optional function for monitoring experiments, add data extraction as needed"""
        return {
            'distance': np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    # ---- ACTION FUNCTIONS ----

    def setup_actions(self, orientation_frame='absolute_orientation'):
        """define action parameters and space"""

        assert orientation_frame in ['absolute_orientation', 'relative_orientation']
        self.orientation_frame = orientation_frame
        self._agent_location = np.array(self.home)
        self._heading_direction = np.array(self.initial_heading)

        # We have 4 possible actions
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)

        if self.orientation_frame == 'absolute_orientation':
            """
            The 4 actions corresponded to "right", "up", "left", "down", but we added more later:
            Extra actions, eg stay in the same location, were added to simulate noise in motion.

            Note: It is possible the extra actions are only taken by
            the environment, and a policy never recommends such!
            
            The following dictionary maps abstract actions from
            `self.action_space` to the direction (changes in
            coordinates) that we will walk in if that action is taken.
            """
            self._action_to_direction = {
                0: np.array([1, 0]),  # move right (increase x)
                1: np.array([0, 1]),  # move up (increase y)
                2: np.array([-1, 0]), # move left
                3: np.array([0, -1]), # move down
                #
                # These were added to simulate noise in motion and its
                # measurement.
                4: np.array([0, 0]),  # stay in same location.
                # go 2 steps in direction you were going!
                5: np.array([2, 0]),
                6: np.array([0, 2]),
                7: np.array([-2, 0]),
                8: np.array([0, -2]),
            }
        else:  # orientation_frame == "relative_orientation"
            """ the 4 actions correspond to turns (right, none, left, turn 180 deg)
          followed by a step in resulting direction """
            self._action_to_rotation = {
                0: np.array([[0, 1], [-1, 0]]),  # turn right 90 deg
                1: np.array([[1, 0], [0, 1]]),  # no turn
                2: np.array([[0, -1], [1, 0]]),  # turn left 90 deg
                3: np.array([[-1, 0], [0, -1]]),  # turn 180 deg (turn around)
            }

    def pick_noisy_action(self, action):
        p = self.motion_noise_prob
        p2 = self.np_random.uniform(0.0, 1.0)
        if p2 > p:
            #print('\n# --------====> NO noise in motion ..\n')
            return action  # no change (perform the intended action)

        #The noisy motion case:
        #print('# picking a random next position!!! %.2f' % p2)
        #print('\n# ***++----> noise in motion!!!\n')
        r = p2 / p
        # When picking a noise movement, some fraction of the time
        # (self.noise_in_direction) either move fwd 2 steps (in the
        # direction of the original action) or stay in the same
        # place. The other portion of the time (of the noise case),
        # pick any action uniformly at random.
        if self.noise_in_direction >= 1 or r < self.noise_in_direction:
            # half the (noise) time, either go fwd 2 steps, or
            # otherwise dont move (stay in original location).
            if self.no_2steps_fwd:  # no possibility of going two steps?
                #print('# ***------> staying in the same place!!')
                return 4  # no change in location, half the noise time
            else:
                if self.np_random.uniform(0.0, 1.0) < 0.5:
                    return 4  # no change in location..
                else:
                    # go 2 steps in the direction you were going.
                    # (if original was 0, you get 5, if 1, then 6, etc..)
                    return action + 5
        # pick an action at random
        a = self.np_random.integers(0, 3, size=1, dtype=int)[0]
        #print('# noise action picked: ', a)
        return a

    def update_location(self, action):
        """Map action to step direction, integrate step, avoid edge and barriers"""

        # To simulate noise in motion, possibly change the action
        # ... (with noise, the agent may stay in the same place at the
        # next time point, or hop two cells forward).
        if self.motion_noise_prob > 0:
            action = self.pick_noisy_action(action)

        direction = self.motion_step(action)

        # make sure agent doesn't leave the grid
        x0, y0 = self._agent_location  # original location
        x, y = np.clip(self._agent_location + direction, 0, self.size - 1)

        if self.env_map[self.barrier, x, y] != 0:  # there is a barrier
            if self.remain_if_barrier:
                x, y = x0, y0  # no change! (remain in old place)
            else:  # pick another (neighboring) location at random..  find
                # alternative not on barrier..
                alt_actions = list(np.arange(self.num_actions))
                while 0 != self.env_map[self.barrier, x, y] and 0 < len(alt_actions):
                    direction = self.motion_step(alt_actions.pop())
                    x, y = np.clip(self._agent_location + direction, 0, self.size - 1)

        self._agent_location = np.array([x, y], dtype=np.int32)
        self._heading_direction = np.array(direction)

    def motion_step(self, action):
        """The increment of motion for a timestep given the action"""
        if self.orientation_frame == 'absolute_orientation':
            return self._action_to_direction[action]
        else:  # orientation_frame == "relative_orientation"
            rot = self._action_to_rotation[action]
            return np.matmul(rot, self._heading_direction)

    # ---- PYGAME AND RENDERING FUNCTIONS ----

    def setup_pygame(self, render_mode):
        """define game parameters"""
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
        self.pygame_size = (512, 512)  # The size of the PyGame window
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.pygame_window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.pygame_window = None
        self.clock = None

    def pointing_head(self):
        """triangle that points in direction of heading_direction"""
        c = (self._agent_location + 0.5) * self.pix_square_size
        delta = 0.5 * self.pix_square_size
        p1 = c + self._heading_direction * delta
        p2 = c + np.matmul(self._action_to_rotation[0], self._heading_direction * delta)
        p3 = c + np.matmul(self._action_to_rotation[2], self._heading_direction * delta)
        return (p1, p2, p3)

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self, force=False):
        """render action and environment using pygame"""
        if (self.pygame_window is None or force) and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.pygame_window = pygame.display.set_mode(self.pygame_size)
            self.pix_square_size = self.pygame_size[0] / self.size
            self.agent_color = (0, 0, 255)
            # ? we will need to redo below anytime the environment changes (e.g., food eaten)
            env_map_image = self.create_env_map_image()
            self.env_map_surf = pygame.image.frombytes(
                env_map_image.tobytes(), self.pygame_size, 'RGB'
            )
            print('# ====----> DONE (in render-frame)\n')
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        # add env map to display
        # ? why does canvas have to be created anew each time?
        canvas = pygame.Surface(self.pygame_size)
        # canvas.fill((255, 255, 255))
        canvas.blit(self.env_map_surf, (0, 0))

        # home square
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                self.pix_square_size * self.home,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        # Now we draw the agent
        if self.orientation_frame == 'absolute_orientation':
            pygame.draw.circle(
                canvas,
                self.agent_color,
                (self._agent_location + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )
        else:  # orientation_frame == "relative_orientation"
            pygame.draw.polygon(canvas, self.agent_color, self.pointing_head())

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.pygame_size[0], self.pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.pygame_size[0]),
                width=3,
            )

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.pygame_window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        self.rgb_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def create_env_map_image(self):
        """make color PIL image from env_map for display"""
        im = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        im[:, :, 0] = self.env_map[self.food, :, :] * 255  # red channel
        im[:, :, 1] = self.env_map[self.barrier, :, :] * 255  # green channel
        im = np.transpose(im, axes=(1, 0, 2))
        # leaving blue channel to show position of agent
        return Image.fromarray(im).resize(self.pygame_size, Image.NEAREST)

    # ---- Auxiliary functions ----

    # For debugging: dump the grid contents around the given x and y
    # with certain radius/width.
    def get_loc_info(self, x=None, y=None, width=2):
        if x is None:
            x, y = self.home
        out, size = '', self.size
        for i in range(x - width, x + width):
            if i < 0 or i >= size:
                continue
            for j in range(y - width, y + width):
                if j < 0 or j >= size:
                    continue
                out += '# x=%d, y=%d, barrier:%d, food:%d\n' % (
                    i,
                    j,
                    self.env_map[self.barrier, i, j],
                    self.env_map[self.food, i, j],
                )
        return out

    def get_agent_location(self):
        return self._agent_location[0], self._agent_location[1]

    # Get the available legal actions at the agent location (those
    # that don't end on a barrier or outside the grid dimensions).
    def get_legal_actions(self):
        x, y = self.get_agent_location()
        legals = []
        if not self.is_barrier_at_loc(x + 1, y):
            legals.append(0)

        if not self.is_barrier_at_loc(x, y + 1):
            legals.append(1)

        if not self.is_barrier_at_loc(x - 1, y):
            legals.append(2)

        if not self.is_barrier_at_loc(x, y - 1):
            legals.append(3)
        return legals

    # Action is a.
    def is_action_legal(self, a):
        x, y = self.get_agent_location()
        
        if a == 0 and not self.is_barrier_at_loc(x + 1, y):
            return True  #  legals.append(0)
        elif a == 1 and not self.is_barrier_at_loc(x, y + 1):
            return True  #  legals.append(1)
        elif a == 2 and not self.is_barrier_at_loc(x - 1, y):
            return True
        elif a == 3 and not self.is_barrier_at_loc(x, y - 1):
            return True
        elif a == 4: # Stay same place?
            return True
        return False

    # Just to test/see locations of barriers (show_barriers or print_barriers)
    # for small envirnoments..
    def barriers_to_text(self):
        rows = ''
        for i in range(self.size):
            s = ''
            for j in range(self.size):
                # Not for text display, j, 1st, then i.
                if self.env_map[self.barrier, j, i]:
                    s += 'b'
                elif self.env_map[self.food,  j, i]:
                    s += 'f'
                else:
                    s += '.'
            rows += s + '\n'
        return rows

    # return a barriers map.
    def gather_barrier_coords(self):
        bar_map = {}
        # include the boundaries too
        for i in range(-1, self.size+1):
            for j in range(-1, self.size+1):
                if i == -1 or i == self.size or j == -1 or j == self.size:
                    bar_map[(i,j)] = True
                    continue
                if self.env_map[self.barrier, i, j]:
                    bar_map[(i,j)] = True
        return bar_map

    # Return true if barrier or walls (end of the grid).
    def is_barrier_at_loc(self, x, y):
        if x < 0 or y < 0:
            return True
        if max(x, y) >= self.size:
            return True
        # print('got here..', x, y, self.env_map[self.barrier, x, y])
        return self.env_map[self.barrier, x, y] >= 1

    # Return true if food is there!
    def is_food_at_loc(self, x, y):
        if x < 0 or y < 0:
            return False
        if max(x, y) >= self.size:
            return False
        return self.env_map[self.food, x, y] >= 1

    def remove_barrier_at_loc(self, x, y):
        self.env_map[self.barrier, x, y] = 0

    # clear foods.
    def remove_foods(self ):
        for x in range(self.size):
            for y in range(self.size):
                self.env_map[self.food, x, y] = 0

    ####

    # Returns None if not next to food.  Otherwise, returns the action
    # that would lead to the location of food.
    def get_action_when_next_to_food(self):
        x, y = self._agent_location # [0]
        if not self.is_barrier_at_loc(x + 1, y):
            if self.env_map[self.food, x+1, y] > 0:
                return 0

        if not self.is_barrier_at_loc(x, y + 1):
            if self.env_map[self.food, x, y+1] > 0:
                return 1

        if not self.is_barrier_at_loc(x - 1, y):
            if self.env_map[self.food, x - 1, y] > 0:
                return 2

        if not self.is_barrier_at_loc(x, y - 1):
            if self.env_map[self.food, x, y - 1] > 0:
                return 3

        return None

    ###

    # Return several closest foods (up to k).
    def get_closest_food(self, x=None, y=None, k=1, insert_if_not=True):
        """get closest food location to x, y:
        minimum |x2-x1| + |y2-y1|  (around x1,y1 coordinates)
        """
        x0, y0 = x, y  # find closest from from x0, y0
        if x is None:
            # Assumes home is center of grid (size/2)
            x0, y0 = self.home  # max radius from home coordinates

        targets = list( self.find_food_closest_to(x0, y0, k) )
        # if x != None or not insert_if_not: # food found?
        if targets != [] or not insert_if_not:  # food found?
            return targets  # food found
        self.place_food(self.place_at_random)  # Try placing food
        return list(self.find_food_closest_to(x0, y0, k))


    def get_food_locs(self):
        foods = []
        for i in range(self.size):
            for j in range(self.size):
                if self.env_map[self.food, i, j] > 0:
                    foods.append( (i,j) )
        return foods
        
    def find_food_closest_to(self, x0, y0, k=1):
        # collect food locs
        foods = self.get_food_locs()
        if foods == []:
            return []
        
        with_dist = []
        for i, j in foods:
            d = abs(x0 - i) + abs(y0 - j)
            with_dist.append(((i,j), d) )

        with_dist.sort(key = lambda x: x[1])
        foods = with_dist[:k]
        return [x[0] for x in foods]
    
    # buggy: expanding rhombuses..
    def find_food_closest_to_old(self, x0, y0, k=1):
        # maximum radius is size (when begin from home/center)
        max_s = 2 * (self.size + 1) # so that rhombus covers all the grid.
        #print('# finding food closest to', x0, y0)
        s = 1
        #print('# max radius: ', max_s)
        targets = set()
        checked = False # 7, 8, food on  14, 0
        while s <= max_s:
            # check for |x|+|y| = s
            for x in range(-s, s + 1):
                y = abs(s - abs(x))  #  y is always positive.
                r = x0 + x  # row
                c = y0 + y  # col
                if r >= self.size or r < 0 or c >= self.size:  # out of grid?
                    continue  # skip

                if r==14 and c==0:
                    print('HERE44', x0, y0)                    
                if r==0 and c==14:
                    print('HERE55', x0, y0)

                if x0==7 and y0==8:
                    print('CHECKING:', 'r:', r, ' c:', c)
                    
                if self.env_map[self.food, r, c] > 0:
                    # print('# *** food at:', r, c, self.env_map[self.food, r, c])
                    targets.add((r, c))
                    k -= 1
                    
                c = y0 - y  # col
                if r==14 and c==0:
                    print('HERE44 ', x0, y0)
                    checked = True
                if r==0 and c==14:
                    print('HERE55', x0, y0)
                    
                if c < 0:
                    continue  # skip

                if x0==7 and y0==8:
                    print('CHECKING:', r, c)
                    
                if self.env_map[self.food, r, c] > 0:
                    # print('# *** food at:', r, c, self.env_map[self.food, r, c])
                    targets.add((r, c))
            if len(targets) >= k:
                return targets
            s += 1

        #if len(targets) == 0:
        # print('\n# HERE11.. s is now:', s, 'checked:', checked, ' start:', x0, y0)
        return targets

    
    def place_food(self, place_at_random=False, x=None, y=None):
        if x is None or y is None:
            x_food, y_food = self.size - 1, self.size - 1
            if self.food_loc is not None:
                x_food, y_food = self.food_loc
            # qprint('\n HERE111', x_food,  y_food)

        if not place_at_random:
            if x is None or y is None:
                x, y = x_food, y_food
            # Explictly remove the barrier at that location.
            self.remove_barrier_at_loc(x, y)
            # print('# HERE88, placed food at:', x, y)
        else:
            # And repeat this until a location without a barrier is
            # found.
            x, y = np.random.randint(x_food), np.random.randint(y_food)

        # Repeat until a location without barrier is found.
        while True:
            if not self.is_barrier_at_loc(x, y):
                #print('\n# **** Placed food at: ', x, y)
                self.env_map[self.food, x, y] = 1  # set food here
                if self.render_mode == 'human':
                    time.sleep(self.sleeper)
                    
                if self.render_mode == 'human':
                    print('# rendering display!\n')
                    self._render_frame(force=True)
                break
            else:
                x, y = np.random.randint(xmax), np.random.randint(ymax)
                #print('\n# trying again')

        #print('# before, targ location:', self._target_location)
        self._target_location = np.array(
            list(
                np.unravel_index(
                    np.argmax(self.env_map[self.food, :, :]),
                    self.env_map[self.food, :, :].shape,
                )
            )
        )
        #print('# after, targ location:', self._target_location, '   requested x,y:', x, y)


    ########


def inside(p, x1, y1, x2, y2):
    """p is inside box, including on edge"""
    x, y = p
    return x1 <= x and x <= x2 and y1 <= y and y <= y2
