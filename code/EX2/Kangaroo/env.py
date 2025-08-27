import sys
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from atari_py.ale_python_interface import ALEInterface

import gym
from torch.multiprocessing import Process, Pipe
from typing import List, Tuple, Union, Optional, Callable, Any

from tianshou.data import Batch
from tianshou.utils import tqdm_config
from tianshou.env import BaseVectorEnv
from tianshou.env.utils import CloudpickleWrapper

from collections import namedtuple

Mask = namedtuple('Mask', 'l t r b')

import matplotlib.pyplot as plt
from argParser import *

from PIL import Image

actionMap = [2, 3, 4]
goal_to_train = [i for i in range(10)]
Num_subgoal = len(goal_to_train)
maxStepsPerEpisode = 500

# location of static objects
# coord_RightDoor = [137, 71]
# coord_MiddleLadder = [80, 112]
# coord_LeftLadder = [24, 157]
# coord_RightLadder = [136, 157]
# coord_Key = [16, 106]
coord_LowLadder = [136,150]
coord_Middleadder = [24,102]
coord_HighLadder = [136,54]
coord_Collection1 = [123,113]
coord_Collection2 = [43,88]
coord_Collection3 = [62,64]
Coord = [[27,159], coord_LowLadder,coord_Middleadder,
         coord_Collection3,coord_Collection1,coord_Collection2, coord_HighLadder,[0, 0]
         ]

# mask_LeftDoor = Mask(18, 51, 26, 91)
# mask_RightDoor = Mask(133, 51, 141, 91)
# mask_MiddleLadder = Mask(72, 93, 88, 137)
# mask_LeftLadder = Mask(16, 138, 32, 182)
# mask_RightLadder = Mask(128, 138, 144, 182)
# mask_Conveyer = Mask(60, 136, 100, 142)
# mask_Chain = Mask(110, 95, 114, 135)
# mask_Key = Mask(13, 97, 19, 115)
mask_LowLadder = [130,128,141,171]
mask_MiddleLadder = [18,80,29,122]
mask_HighLadder = [130,21,141,75]
mask_Collection1 = [118,107,126,119]
mask_Collection2 = [38,83,46,95]
mask_Collection3 = [58,59,66,71]
loclist = ['Man', 'LowLadder', 'MiddleLadder', 'Collection3', 'Collection1', 'Collection2', 'HighLadder']

idx2loc, loc2idx = {}, {}

for i in range(len(loclist)):
    idx2loc[i] = loclist[i]

for key, value in idx2loc.items():
    loc2idx[value] = key

idx2predicate = {0: 'ActorOnSpot', 1: 'ActorWithObject', 2: 'ActorWithOutObject', 3: 'PathExist', 4: 'Conditional', 5: 'Near',
                 6: 'Away'}

predicate2idx = {}
for key, value in idx2predicate.items():
    predicate2idx[value] = key

from collections import deque


def initialize_ale_env(rom_file, args):
    ale = ALEInterface()
    if args.display_screen:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False)
        elif sys.platform.startswith('linux'):
            ale.setBool('sound', False)
        ale.setBool('display_screen', False)
    ale.setInt('frame_skip', args.frame_skip)
    ale.setFloat('repeat_action_probability', 0.0)
    ale.setBool('color_averaging', args.color_averaging)

    ale.setInt('random_seed', 0)  # hoang addition to fix the random seed across all environment
    ale.loadROM(rom_file)

    if args.minimal_action_set:
        actions = ale.getMinimalActionSet()
    else:
        actions = ale.getLegalActionSet()
    path = '/root/xxx-master/Game/image/'
    img = ale.getScreenRGB()
    img = img[:,:,[2,1,0]]
    # cv2.imwrite(path + '0' + '.jpg', img)
    # act = 0
    # for i in range(6):
    #     act = actionMap[i+2]
    #     for j in range(50):
    #         ale.act(act)
    #         ale.saveScreenPNG('/root/hrlid-master/Game/image/'+str(act)+'-'+str(j)+'.png')
    # ale.saveScreenPNG('/root/hrlid-master/Game/image/m.png')
    return ale, actions


class ALEEnvironment():

    def __init__(self, rom_file, args):
        self.ale, self.actions = initialize_ale_env(rom_file, args)
        self.histLen = 4
        self.screen_width = args.screen_width
        self.screen_height = args.screen_height

        self.restart()
        self.agentOriginLoc = [25,165]
        self.agentLastX = 25
        self.agentLastY = 165

        self.devilLastX = 0
        self.devilLastY = 0

        self.reachedGoal = np.zeros((10, 10))
        self.last_spot = 'Man'
        self.mode = 'train'

    def getScreen(self, ScreenType='Gray'):
        if ScreenType == 'Gray':
            screen = self.ale.getScreenGrayscale()
        else:
            screen = self.ale.getScreenRGB()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized

    def getState(self):
        return np.reshape(self.getScreen(), (1, self.screen_width, self.screen_height))

    def initializeHistState(self):
        self.histState = np.zeros((self.histLen, self.screen_width, self.screen_height))
        initialState = self.getState()[:, :, 0]
        for num in range(self.histLen):
            self.histState[num, :, :] = initialState

    def restart(self):
        self.ale.reset_game()
        self.life_lost = False
        # for _ in range(200):
        #     rew = self.ale.act(3)
        self.initializeHistState()

        self.initializeScreen = self.ale.getScreenRGB()

        # initialize logcY buf
        self.agent_locY_buf = deque(maxlen=5)
        self.agent_locY_buf.append(161)

    @property
    def numActions(self):
        return len(self.actions)

    def getLoc(self, img, obj='Man'):

        # img : RGB Array
        if obj == 'Skull':
            color = [236, 236, 236]
            mean_x = self.devilLastX
            mean_y = self.devilLastY
        else:
            # color = [200, 72, 72]
            color = [223,183,85]
            mean_x = self.agentLastX
            mean_y = self.agentLastY

        mask = np.zeros(np.shape(img))
        mask[:, :, 0] = color[0]
        mask[:, :, 1] = color[1]
        mask[:, :, 2] = color[2]

        diff = img - mask
        indxs = np.where(diff == 0)

        if (np.shape(indxs[0])[0]):
            y = indxs[0][-1]
            x = indxs[1][-1]
            if obj == 'Man':
                if y >= 28:
                    self.agentLastX, self.agentLastY = x - 3, y - 3
                    return [x - 3, y - 6]
                else:
                    color = [187,159,71]
                    mean_x = self.agentLastX
                    mean_y = self.agentLastY
                    mask = np.zeros(np.shape(img))
                    mask[:, :, 0] = color[0]
                    mask[:, :, 1] = color[1]
                    mask[:, :, 2] = color[2]
                    diff = img - mask
                    indxs = np.where(diff == 0)
                    if (np.shape(indxs[0])[0]):
                        y = indxs[0][-1]
                        x = indxs[1][-1]
                        if y >= 28:
                            self.agentLastX, self.agentLastY = x - 3, y - 3
                            return [x - 3, y - 3]
                        else:
                            x, y = self.agentLastX, self.agentLastY
                            return [x, y]
                    else:
                        # 这里可能不太严谨
                        if obj == 'Man':
                            x, y = self.agentLastX, self.agentLastY
                            return [x, y]
                        else:
                            x, y = None, None
                            return [x, y]
            else:

                if y > 150:
                    self.devilLastX, self.devilLastY = x, y
                    return [x - 1, y - 6]
                else:
                    return [None, None]
        else:
            #这里可能不太严谨
            if obj == 'Man':
                x, y = self.agentLastX, self.agentLastY
                return [x, y]
            else:
                x, y = None, None
                return [x, y]

    def distanceReward(self, lastgoal, goal):
        if (lastgoal == None):
            lastgoal_x, lastgoal_y = self.agentOriginLoc
        else:
            lastgoal_x, lastgoal_y = Coord[lastgoal]
        img = self.ale.getScreenRGB()
        man_x, man_y = self.getLoc(img, 'Man')

        goal_x, goal_y = Coord[goal]

        if goal_x == None or goal_y == None:
            return 0
        dis = np.sqrt(((man_x - goal_x) ** 2)*2)
        disLast = np.sqrt((man_x - lastgoal_x) ** 2 + (man_y - lastgoal_y) ** 2)
        disGoals = np.sqrt((goal_x - lastgoal_x) ** 2 + (goal_y - lastgoal_y) ** 2)
        # if goal>1:
        #     return 0.001 *  (-1)*dis+0.004*high_dis
        # else:
        if goal == 3 or goal == 4:
            high_dis = np.sqrt(((man_y - goal_y) ** 2))
            if man_y > goal_y:
                high_dis = (-1) * high_dis
            return 0.002 * high_dis
        else:
            return 0.001 * (-1) * dis
        # return 0.001 *  (-1)*dis+0.001*high_dis

    def getStackedState(self):

        return self.histState

    def isGameOver(self):
        return self.ale.game_over()

    def isTerminal(self):

        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()

    def isLifeLost(self):
        return self.life_lost

    def skullExist(self, img):

        skull_x, skull_y = self.getLoc(img, 'Skull')
        if skull_y == None:
            return False
        return True

    def keyExist(self):

        img = self.ale.getScreenRGB()

        mask = np.ones((2, 1, 3))
        mask[:, :, 0] = 214
        mask[:, :, 1] = 92
        mask[:, :, 2] = 92

        if (img[111:113, 124:125, :] == mask).all() and (img[87:89, 44:45, :] == mask).all() and (img[63:65, 64:65, :] == mask).all():
            return True
        return False
    def Collection1Exist(self):
        img = self.ale.getScreenRGB()
        mask = np.ones((1, 1, 3))
        mask[:, :, 0] = 214
        mask[:, :, 1] = 92
        mask[:, :, 2] = 92
        if (img[111:112,124:125,:]==mask).all():
            return True
        return False
    def Collection3Exist(self):
        img = self.ale.getScreenRGB()
        mask = np.ones((1, 1, 3))
        mask[:, :, 0] = 214
        mask[:, :, 1] = 92
        mask[:, :, 2] = 92
        if (img[63:64,62:63,:]==mask).all():
            return True
        return False
    def get_symbolic_tensor(self):

        '''
            change symbolic state into array
            the predicate of each channel :
            0 : ActorOnSpot
            1 : ActorWithObject
            2 : ActorWithoutObject
            3 : PathExist
            4 : Conditional
loclist = ['Man', 'LowLadder', 'MiddleLadder', 'Collection3', 'Collection1', 'Collection2', 'HighLadder']
            0        1              2             3              4
            the object order of each dimension
            0 : Man
            1 : RightDoor
            2 : Key
            3 : MiddleLadder
            4 : LeftLadder
            5 : RightLadder
            6 : SkullLeft
        '''

        img = self.ale.getScreenRGB()
        self.state_tensor = np.zeros((5, 5, 5))
        symbolic_state = set()

        x, y = self.getLoc(img, 'Man')
        a, b = self.getLoc(img, 'Skull')

        if self.Collection1Exist() and self.reach_goal() != 4:
            self.state_tensor[2][0][4] = 1
            symbolic_state.add(('ActorWithOutObject'))
        else:
            self.state_tensor[1][0][4] = 1
            symbolic_state.add(('ActorWithObject'))
        if self.Collection3Exist() and self.reach_goal() != 3:
            self.state_tensor[2][0][3] = 1
            symbolic_state.add(('ActorWithOutObject'))
        else:
            self.state_tensor[1][0][3] = 1
            symbolic_state.add(('ActorWithObject'))

        spot = self.ActorOnSpot(img, (x, y))
        self.state_tensor[0][0][loc2idx[spot]] = 1
        symbolic_state.add(('ActorOnSpot', spot))

        # connected_tuple = [(3, 5), (5, 6), (6, 4), (4, 2), (3, 1)]
        connected_tuple = [(0, 1), (1, 4), (4, 2), (2, 3)]

        for ple in connected_tuple:
            self.state_tensor[-2][ple[0]][ple[1]] = 1
            self.state_tensor[-2][ple[1]][ple[0]] = 1

        self.state_tensor[-1][2][1] = 1

        self.symbolic_state = symbolic_state

        return self.state_tensor.reshape((5, 5 ** 2))

    def action_to_symbolic(self, action):#action不确定是不是这些 能对的上的

        # actionMap = {0: 0,
        #              1: 1,
        #              2: 2,
        #              3: 3,
        #              4: 4,
        #              5: 5,
        #              6: 11,
        #              7: 12}
        # actionMap = {0: 2,
        #              1: 3,
        #              2: 4,
        #              3: 5}
        # actionExplain = {2: 'up',
        #                  3: 'right',
        #                  4: 'left',
        #                  5:'down'}
        actionMap = {1: 2,
                     2: 3,
                     3: 4}
        actionExplain = {2: 'up',
                         3: 'right',
                         4: 'left'}

        return actionExplain[actionMap[action]]

    def create_dynamic_object_mask(self, img, coord_man, coord_skull):

        man_x, man_y = coord_man
        skull_x, skull_y = coord_skull

        self.mask_man = Mask(man_x - 4, man_y - 10, man_x + 4, man_y + 10)

        if skull_x != None:
            self.mask_skull = Mask(skull_x - 4, skull_y - 6, skull_x + 4, skull_y + 6)
        else:
            self.mask_skull = Mask(None, None, None, None)

        return self.mask_man, self.mask_skull

    def draw_mask(self, img):

        coord_man = self.getLoc(img, 'Man')
        coord_skull = self.getLoc(img, 'Skull')
        mask_Man, mask_Skull = self.create_dynamic_object_mask(img, coord_man, coord_skull)
        mask_list = [mask_LeftDoor, mask_RightDoor, mask_MiddleLadder,
                     mask_LeftLadder, mask_RightLadder, mask_Conveyer,
                     mask_Chain, mask_Man]
        if mask_Skull.l != None:
            mask_list.append(mask_Skull)
        if self.keyExist():
            mask_list.append(mask_Key)
        for mask in mask_list:
            cv2.rectangle(img, (mask.l, mask.t), (mask.r, mask.b), (0, 0, 255), 1)
        return img

    def select_goal(self, goal, reach_goal):

        '''
        0 : Man
        1 : RightDoor
        2 : Key
        3 : MiddleLadder
        4 : LeftLadder
        5 : RightLadder
        6 : Skull
        '''

        if reach_goal == 2:
            if goal == 4:
                return 3

        if reach_goal == 3 and self.keyExist():
            if goal == 5:
                return 0

        if reach_goal == 3 and not self.keyExist():
            if goal == 1:
                return 6

        if reach_goal == 4:
            if goal == 5:
                return 4

        if reach_goal == 5:

            if goal == 3 and not self.keyExist():
                return 5

            if goal == 6 and self.keyExist():
                return 1

        if reach_goal == 6:
            if goal == 2:
                return 2
        return -1
    def select_goal_(self, goal, reach_goal):
        '''
        修改
        loclist = ['Man', 'LowLadder', 'MiddleLadder', 'HighLadder', 'Collection1', 'Collection2', 'Collection3']
                    0        1              2             3              4                 5            6
        0 : Man
        1 : RightDoor
        2 : Key
        3 : MiddleLadder
        4 : LeftLadder
        5 : RightLadder
        6 : Skull
        '''
        if reach_goal == 0:
            if goal == 1:
                return 0
        if reach_goal == 1:
            if goal == 4:
                return 1
        if reach_goal == 4:
            if goal == 2:
                return 2
        if reach_goal == 2:
            if goal == 3:
                return 3
        return -1

    def output_man_loc(self):
        img = self.ale.getScreenRGB()
        man_x, man_y = self.getLoc(img, 'Man')

    def reach_goal(self):
        goal_reached = -1
        img = self.ale.getScreenRGB()
        man_x, man_y = self.getLoc(img, 'Man')
        for idx in [0,1,4,2,3]:#修改
            goal = idx2loc[idx]
            if self.goalReached(img, goal, (man_x, man_y)):
                goal_reached = idx
                break
        return goal_reached

    def reach_goal_(self):
        goal_reached = -1
        img = self.ale.getScreenRGB()
        man_x, man_y = self.getLoc(img, 'Man')
        for idx in [0,1,4,2,3]:#修改
            goal = idx2loc[idx]
            if self.goalReached_(img, goal, (man_x, man_y)):
                goal_reached = idx
                break
        return goal_reached

    def goalReached(self, img, loc, coords=None):

        if self.isTerminal():
            return False

        if coords == None:
            man_x, man_y = self.getLoc(img, 'Man')
        else:
            man_x, man_y = coords
        if loc == 'Man':
            if man_x >= 20 and man_x <= 33 and man_y >= 145 and man_y <= 172:
                return True
        if loc == 'LowLadder':
            if man_x >= 126 and man_x <= 143 and man_y >= 128 and man_y <= 172:
                return True
        if loc == 'MiddleLadder':
            if man_x >= 16 and man_x <= 33 and man_y >= 80 and man_y <= 123:
                return True
        if loc == 'HighLadder':
            if man_x >= 130 and man_x <= 143 and man_y >= 32 and man_y <= 75:
                return True
        if loc == 'Collection1':
            if man_x >= 118 and man_x <= 126 and man_y >= 100 and man_y <= 123:
                return True
        if loc == 'Collection3':
            if man_x >= 54 and man_x <= 69 and man_y >= 55 and man_y <= 75:
                return True
        return False

    def goalReached_(self, img, loc, coords=None):

        if self.isTerminal():
            return False

        if coords == None:
            man_x, man_y = self.getLoc(img, 'Man')
        else:
            man_x, man_y = coords
        if loc == 'Man':
            if man_x >= 20 and man_x <= 33 and man_y >= 145 and man_y <= 172:
                return True
        if loc == 'LowLadder':
            if man_x >= 126 and man_x <= 143 and man_y >= 128 and man_y <= 172:
                return True
        if loc == 'MiddleLadder':
            if man_x >= 16 and man_x <= 29 and man_y >= 80 and man_y <= 123:
                return True
        if loc == 'HighLadder':
            if man_x >= 130 and man_x <= 143 and man_y >= 32 and man_y <= 75:
                return True
        if loc == 'Collection1':
            if man_x >= 112 and man_x <= 131 and man_y >= 99 and man_y <= 123:
                return True
        if loc == 'Collection3':
            if man_x >= 54 and man_x <= 69 and man_y >= 55 and man_y <= 75:
                return True
        return False

    def ActorOnSpot(self, img, coords=None):

        if coords == None:
            man_x, man_y = self.getLoc(img, 'Man')
        else:
            man_x, man_y = coords

        spot = None
        for loc in loclist[1:]:
            if self.goalReached(img, loc, (man_x, man_y)):
                spot = loc
                self.last_spot = spot
                return spot
        return self.last_spot

    def step(self, action, lastgoal, goal):

        reward = self.ale.act(self.actions[action])
        reward = self.distanceReward(lastgoal, goal)

        if self.isLifeLost():
            done = True
        return self.histState, reward, done, {}

    def act(self, action):
        lives = self.ale.lives()
        reward = self.ale.act(self.actions[action])
        self.life_lost = (not lives == self.ale.lives())
        curState = self.getState()

        img = self.ale.getScreenRGB()
        x, y = self.getLoc(img)
        spot = self.ActorOnSpot(img, (x, y))

        '''
        x,y = self.getLoc(self.ale.getScreenRGB())
        self.agent_locY_buf.append(y)
        '''

        self.histState = np.concatenate((self.histState[1:, :, :], curState), axis=0)
        return reward

    def beginNextLife(self):
        self.life_lost = False
        for _ in range(19):
            rew = self.act(0)
        self.initializeHistState()

    def Near(self, object_loc1, object_loc2, idx1=None, idx2=None):

        x1, y1 = object_loc1
        x2, y2 = object_loc2

        if x1 == None or x2 == None or y1 == None or y2 == None:
            return 'None'

        if (idx1 == 0 and idx2 == 9) or (idx1 == 9 and idx2 == 0):
            if abs(x1 - x2) <= 20 and abs(y1 - y2) <= 3:
                '''
                for i in self.agent_locY_buf:
                    if i != 169 or i != 170:
                        return 'Away'
                '''
                return 'Near'
            return 'Away'

        if abs(x1 - x2) < 20 and abs(y1 - y2) < 33:
            return 'Near'
        return 'Away'
