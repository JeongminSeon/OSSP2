import random
import pygame
import time
import sys
from pygame.locals import *
from QuoridorEnv import QuoridorEnv
from DumbAgent import DumbAgent
from MctsAgent import *

#  QuoridorEnv와 동기화가 필요한 항목들
AGENT_1 = 100
AGENT_2 = 200
ACT_MOVE_CNT = 4
ACT_MOVE_NORTH = 0
ACT_MOVE_WEST = 1
ACT_MOVE_SOUTH = 2
ACT_MOVE_EAST = 3

FPS = 30  # frames per second, the general speed of the program

REVEALSPEED = 8  # speed boxes' sliding reveals and covers

#         R    G    B
GRAY = (100, 100, 100)
DARKGRAY = (30, 30, 30)
NAVYBLUE = (60,  60, 100)
WHITE = (255, 255, 255)
RED = (255,   0,   0)
GREEN = (0, 255,   0)
BLUE = (0,   0, 255)
YELLOW = (255, 255,   0)
ORANGE = (255, 128,   0)
PURPLE = (255,   0, 255)
CYAN = (0, 255, 255)

BGCOLOR = NAVYBLUE
LIGHTBGCOLOR = GRAY
BOXCOLOR = WHITE
HIGHLIGHTCOLOR = BLUE

DONUT = 'donut'
SQUARE = 'square'
DIAMOND = 'diamond'
LINES = 'lines'
OVAL = 'oval'


class QuoridorGUI():
    def __init__(self, env, agent_num1=-1, agent_num2=-1, auto_pilot1=None, auto_pilot2=None):
        self.input_action = [-1, -1]
        pygame.init()
        self.BOXSIZE = 40  # size of box height & width in pixels
        self.GAPSIZE = 15  # size of gap between boxes in pixels
        self.width = env.width
        self.env = env
        self.game_done = False
        self.XMARGIN = 60
        self.YMARGIN = 40
        self.EXTRAWIDTH = 200
        self.TEXTMARGIN = 100
        self.WINDOWWIDTH = (self.BOXSIZE + self.GAPSIZE) * \
            self.width + self.XMARGIN * 2 - self.GAPSIZE + self.EXTRAWIDTH
        self.WINDOWHEIGHT = (self.BOXSIZE + self.GAPSIZE) * \
            self.width + self.YMARGIN * 2 - self.GAPSIZE
        self.FPSCLOCK = pygame.time.Clock()
        self.DISPLAYSURF = pygame.display.set_mode(
            (self.WINDOWWIDTH, self.WINDOWHEIGHT))
        self.player1_on = False
        self.player2_on = False
        self.player1_num = -1
        self.player2_num = -1
        self.auto_pilot1 = None
        self.auto_pilot2 = None
        if agent_num1 != -1:
            if env.agNumList[0] == agent_num1:
                self.player1_on = True
                self.player1_num = agent_num1
            elif env.agNumList[1] == agent_num1:
                self.player2_on = True
                self.player2_num = agent_num1
            else:
                raise Exception('QuoridorGUI 초기화 에러!\n잘못된 agent_num1 전달')
        if agent_num2 != -1:
            if env.agNumList[0] == agent_num2:
                self.player1_on = True
                self.player1_num = agent_num2
            elif env.agNumList[1] == agent_num2:
                self.player2_on = True
                self.player2_num = agent_num2
            else:
                raise Exception('QuoridorGUI 초기화 에러!\n잘못된 agent_num2 전달')
        if self.player1_on and self.player2_on:
            if env.last_played == self.player1_num:  # agent2 가 다음 플레이어일 경우
                self.legal_action = env.get_legal_action(
                    env.get_state(self.player2_num))
            else:
                self.legal_action = env.get_legal_action(
                    env.get_state(self.player1_num))
        elif self.player1_on:
            self.legal_action = env.get_legal_action(
                env.get_state(self.player1_num))
            if auto_pilot1 == None:
                raise Exception('GUI 초기화 에러! 플레이어 1명이지만, 에이전트 전달하지 않음.')
            self.auto_pilot2 = auto_pilot1
        elif self.player2_on:
            self.legal_action = env.get_legal_action(
                env.get_state(self.player2_num))
            if auto_pilot1 == None:
                raise Exception('GUI 초기화 에러! 플레이어 1명이지만, 에이전트 전달하지 않음.')
            self.auto_pilot1 = auto_pilot1
        else:
            if auto_pilot1 == None or auto_pilot2 == None:
                raise Exception('GUI 초기화 에러! 플레이어 0명이지만, 에이전트 2개 미만 전달.')
            self.auto_pilot1 = auto_pilot1
            self.auto_pilot2 = auto_pilot2
            self.legal_action = None
        self.mousex = 0  # used to store x coordinate of mouse event
        self.mousey = 0  # used to store y coordinate of mouse event
        self.oldMousex = 0  # used to store x coordinate of mouse event
        self.oldMousey = 0  # used to store y coordinate of mouse event
        self.mouseDown = False
        self.mouseUp = False
        pygame.display.set_caption('QUORIDOR')

        self.player_map = self.getPlayerMap(env)
        self.wall_map = env.map
        self.DISPLAYSURF.fill(BGCOLOR)
        self.FONTSIZE = 20
        self.font = pygame.font.SysFont("arial", self.FONTSIZE, True)

    def startGame(self):
        while not self.game_done:  # main game loop
            auto_pilot_played = False
            self.mouseUp = False
            self.drawBoard()

            # PRINT WHICH TURN
            pygame.draw.rect(self.DISPLAYSURF, (230, 230, 230), [(self.BOXSIZE + self.GAPSIZE) *
                             self.width + self.XMARGIN * 2 - self.GAPSIZE - 10, 0, self.EXTRAWIDTH+self.XMARGIN * 0.6 + 10, self.WINDOWHEIGHT])
            text0 = self.font.render(
                "TURN: ", True, (0, 0, 0))
            self.DISPLAYSURF.blit(text0, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE, 10))
            if (self.env.get_last_played() == AGENT_1):
                text1 = self.font.render(
                    "BLUE", True, BLUE)
            else:
                text1 = self.font.render(
                    "RED", True, RED)
            self.DISPLAYSURF.blit(text1, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE+self.FONTSIZE*3.5, 10))

            # P1 의 status 문자 출력
            p1_text0 = self.font.render(
                "RED: ", True, RED)
            self.DISPLAYSURF.blit(p1_text0, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE, self.YMARGIN+self.FONTSIZE*2))
            p1_text1 = self.font.render(
                "Wall: "+str(self.env.get_state(AGENT_1)[1][0][2]), True, DARKGRAY)
            self.DISPLAYSURF.blit(p1_text1, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE+self.FONTSIZE*5, self.YMARGIN+self.FONTSIZE*2))
            p1_text2 = self.font.render(
                "move: "+str(self.env.get_move_count()[0]), True, DARKGRAY)
            self.DISPLAYSURF.blit(p1_text2, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE+self.FONTSIZE*5, self.YMARGIN+self.FONTSIZE*4))

            # P2 의 status 문자 출력
            p2_text0 = self.font.render(
                "BLUE: ", True, BLUE)
            self.DISPLAYSURF.blit(p2_text0, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE, self.YMARGIN+self.FONTSIZE*8))
            p2_text1 = self.font.render(
                "Wall: "+str(self.env.get_state(AGENT_1)[1][1][2]), True, DARKGRAY)
            self.DISPLAYSURF.blit(p2_text1, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE+self.FONTSIZE*5, self.YMARGIN+self.FONTSIZE*8))
            p2_text2 = self.font.render(
                "move: "+str(self.env.get_move_count()[1]), True, DARKGRAY)
            self.DISPLAYSURF.blit(p2_text2, ((
                self.BOXSIZE + self.GAPSIZE) * self.width + self.XMARGIN * 2 - self.GAPSIZE+self.FONTSIZE*5, self.YMARGIN+self.FONTSIZE*10))
            # if self.player1_on or self.player2_on:
            for event in pygame.event.get():  # event handling loop
                if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEMOTION:
                    if not self.mouseDown:
                        self.oldMousex = self.mousex
                        self.oldMousey = self.mousey
                    self.mousex, self.mousey = event.pos
                elif event.type == MOUSEBUTTONDOWN:
                    self.oldMousex, self.oldMousey = event.pos
                    self.mouseDown = True
                elif event.type == MOUSEBUTTONUP:
                    # if not self.mouseDown:
                    #     self.oldMousex = self.mousex
                    #     self.oldMousey = self.mousey
                    self.mousex, self.mousey = event.pos
                    self.mouseUp = True
                    self.mouseDown = False
            if (self.auto_pilot1 != None):
                if (self.env.get_last_played() != self.auto_pilot1.get_agent_num()):
                    move = self.auto_pilot1.get_action()
                    pygame.time.wait(300)
                    if move != None:
                        state, reward, done = self.env.step(
                            self.auto_pilot1.get_agent_num(), move)
                        self.game_done = done
                        auto_pilot_played = True
                    else:
                        raise Exception('GUI: auto pilot1 구동 실패')
            if (self.auto_pilot2 != None and not auto_pilot_played):
                if (self.env.get_last_played() != self.auto_pilot2.get_agent_num()):
                    move = self.auto_pilot2.get_action()
                    pygame.time.wait(300)
                    if move != None:
                        state, reward, done = self.env.step(
                            self.auto_pilot2.get_agent_num(), move)
                        self.game_done = done
                    else:
                        raise Exception('GUI: auto pilot2 구동 실패')
            if self.env.ask_state_changed():
                self.env.set_state_changed_false()
                self.updateLegalActionAndState()
            boxx, boxy, third_element = self.getObjAtPixel(
                self.oldMousex, self.oldMousey, self.mousex, self.mousey)
            if third_element != -1:
                # if cursor is on legal object
                if self.checkObjLegalAndGetMove(boxx, boxy, third_element):
                    # The mouse is currently over a empty_wall.
                    if third_element == 1 or third_element == 2:
                        self.drawHighlightWall(boxx, boxy, third_element)
                    # The mouse is currently over a box.
                    elif third_element == 0:
                        self.drawHighlightBox(boxx, boxy)
                    # 착수
                    if self.mouseUp:
                        self.confirmMove()
                        self.env.set_state_changed_false()
                        self.updateLegalActionAndState()
            pygame.display.update()
            self.FPSCLOCK.tick(FPS)
        self.gameWonAnimation()

    def generateStartingPos(self, val):
        player_map = []
        for i in range(self.width):
            player_map.append([val] * self.width)
        return player_map

    def leftTopCoordsOfBox(self, boxx, boxy):
        # Convert board coordinates to pixel coordinates
        left = boxx * (self.BOXSIZE + self.GAPSIZE) + self.XMARGIN
        top = boxy * \
            (self.BOXSIZE + self.GAPSIZE) + self.YMARGIN
        return (left, top)

    def leftTopCoordsOfBoxFlipped(self, boxx, boxy):
        # Convert board coordinates to pixel coordinates
        left = boxx * (self.BOXSIZE + self.GAPSIZE) + self.XMARGIN
        top = (self.width - boxy - 1) * \
            (self.BOXSIZE + self.GAPSIZE) + self.YMARGIN
        return (left, top)

    def leftTopCoordsOfWallCenter(self, boxx, boxy):
        # Convert board coordinates to pixel coordinates
        left = boxx * (self.BOXSIZE + self.GAPSIZE) + \
            self.XMARGIN + self.BOXSIZE
        top = boxy * (self.BOXSIZE +
                      self.GAPSIZE) + self.YMARGIN + self.BOXSIZE
        return (left, top)

    def leftTopCoordsOfWallCenterFlipped(self, boxx, boxy):
        # Convert board coordinates to pixel coordinates
        left = boxx * (self.BOXSIZE + self.GAPSIZE) + \
            self.XMARGIN + self.BOXSIZE
        top = (self.width - boxy - 1) * (self.BOXSIZE +
                                         self.GAPSIZE) + self.YMARGIN + self.BOXSIZE
        return (left, top)

    def getObjAtPixel(self, oldx, oldy, newx, newy):
        dir_left = False
        dir_up = False
        vertical_is_big = False
        third_element = 0
        if ((oldx - newx) > 0):
            dir_left = True
        if ((oldy - newy) > 0):
            dir_up = True
        a = oldx - newx
        b = oldy - newy
        if a < 0:
            a *= -1
        if b < 0:
            b *= -1
        if a < b:
            vertical_is_big = True
        x = newx - self.XMARGIN
        x_a = x // (self.BOXSIZE + self.GAPSIZE)
        x_b = x % (self.BOXSIZE + self.GAPSIZE)
        x_c = False
        y = newy - self.YMARGIN
        y_a = y // (self.BOXSIZE + self.GAPSIZE)
        y_b = y % (self.BOXSIZE + self.GAPSIZE)
        y_c = False
        if x < 0 or y < 0 or x_a > self.width - 1 or y_a > self.width - 1:
            return (0, 0, -1)
        if x_b > self.BOXSIZE:
            x_c = True
        if y_b > self.BOXSIZE:
            y_c = True

        if x_c:
            if y_c:
                if vertical_is_big:
                    third_element = 2
                    if not dir_up:
                        y_a -= 1
                else:
                    third_element = 1
                    if not dir_left:
                        x_a -= 1
                if x_a < 0 or y_a < 0 or x_a >= self.width - 1 or y_a >= self.width - 1:
                    return (0, 0, -1)  # 테두리에 위치하는 상황
                else:
                    return (x_a, y_a, third_element)

            else:
                third_element = 2
                if not dir_up:
                    y_a -= 1
                if x_a < 0 or y_a < 0 or x_a >= self.width - 1 or y_a >= self.width - 1:
                    return (0, 0, -1)  # 테두리에 위치하는 상황
                else:
                    return (x_a, y_a, third_element)
        elif y_c:
            third_element = 1
            if not dir_left:
                x_a -= 1
            if x_a < 0 or y_a < 0 or x_a >= self.width - 1 or y_a >= self.width - 1:
                return (0, 0, -1)  # 테두리에 위치하는 상황
            else:
                return (x_a, y_a, third_element)
        else:
            return (x_a, y_a, 0)  # 말의 다음 위치

    def drawIcon(self, shape, color, boxx, boxy):
        quarter = int(self.BOXSIZE * 0.25)  # syntactic sugar
        half = int(self.BOXSIZE * 0.5)  # syntactic sugar

        # get pixel coords from board coords
        left, top = leftTopCoordsOfBoxFlipped(boxx, boxy)
        # Draw the shapes
        if shape == DONUT:
            pygame.draw.circle(self.DISPLAYSURF, color,
                               (left + half, top + half), half - 5)
            pygame.draw.circle(self.DISPLAYSURF, BGCOLOR,
                               (left + half, top + half), quarter - 5)
        elif shape == SQUARE:
            pygame.draw.rect(self.DISPLAYSURF, color, (left + quarter,
                                                       top + quarter, self.BOXSIZE - half, self.BOXSIZE - half))
        elif shape == DIAMOND:
            pygame.draw.polygon(self.DISPLAYSURF, color, ((left + half, top), (left + self.BOXSIZE - 1,
                                top + half), (left + half, top + self.BOXSIZE - 1), (left, top + half)))
        elif shape == LINES:
            for i in range(0, self.BOXSIZE, 4):
                pygame.draw.line(self.DISPLAYSURF, color,
                                 (left, top + i), (left + i, top))
                pygame.draw.line(self.DISPLAYSURF, color, (left + i,
                                                           top + self.BOXSIZE - 1), (left + self.BOXSIZE - 1, top + i))
        elif shape == OVAL:
            pygame.draw.ellipse(self.DISPLAYSURF, color,
                                (left, top + quarter, self.BOXSIZE, half))

    def getPlayerMap(self, env):
        ret = []
        for x in range(env.width):
            ret.append([0] * env.width)
        ret[env.player_status[0][0]][env.player_status[0][1]] = 1
        ret[env.player_status[1][0]][env.player_status[1][1]] = 2
        # 둘이 같은 위치에 있을 경우
        if (env.player_status[0][0] == env.player_status[1][0] and env.player_status[0][1] == env.player_status[1][1]):
            ret[env.player_status[1][0]][env.player_status[1][1]] = 3
        return ret

    def drawBoard(self):
        self.DISPLAYSURF.fill(BGCOLOR)  # drawing the window
        # player가 설 수 있는 위치들을 그림
        for boxx in range(self.width):
            for boxy in range(self.width):
                left, top = self.leftTopCoordsOfBox(
                    boxx, self.width - 1 - boxy)
                pygame.draw.rect(self.DISPLAYSURF, GRAY,
                                 (left, top, self.BOXSIZE, self.BOXSIZE))
                if self.player_map[boxx][boxy] == 1:
                    pygame.draw.rect(self.DISPLAYSURF, RED,
                                     (left + 5, top + 5, self.BOXSIZE - 10, self.BOXSIZE - 10))
                elif self.player_map[boxx][boxy] == 2:
                    pygame.draw.rect(self.DISPLAYSURF, BLUE,
                                     (left + 5, top + 5, self.BOXSIZE - 10, self.BOXSIZE - 10))
                elif self.player_map[boxx][boxy] == 3:
                    pygame.draw.rect(self.DISPLAYSURF, BLUE,
                                     (left + 5, top + 5, self.BOXSIZE - 10, self.BOXSIZE / 2 - 5))
                    pygame.draw.rect(self.DISPLAYSURF, RED,
                                     (left + 5, top+self.BOXSIZE / 2, self.BOXSIZE - 10, self.BOXSIZE / 2 - 5))
        # wall들을 그림
        for boxx in range(self.width - 1):
            for boxy in range(self.width - 1):
                if self.wall_map[0][boxx][boxy]:
                    left = self.XMARGIN + (self.BOXSIZE + self.GAPSIZE) * boxx
                    top = self.YMARGIN + (self.BOXSIZE + self.GAPSIZE) * \
                        (self.width - 2 - boxy) + self.BOXSIZE
                    pygame.draw.rect(self.DISPLAYSURF, ORANGE, (left + 5,
                                                                top + 5, self.BOXSIZE + self.GAPSIZE + self.BOXSIZE - 10, self.GAPSIZE - 10), 0)
                if self.wall_map[1][boxx][boxy]:
                    left = self.XMARGIN + \
                        (self.BOXSIZE + self.GAPSIZE) * boxx + self.BOXSIZE
                    top = self.YMARGIN + \
                        (self.BOXSIZE + self.GAPSIZE) * (self.width - boxy - 2)
                    pygame.draw.rect(self.DISPLAYSURF, ORANGE, (left + 5,
                                                                top + 5, self.GAPSIZE - 10, self.BOXSIZE + self.GAPSIZE + self.BOXSIZE - 10), 0)

    def drawHighlightBox(self, boxx, boxy):
        left, top = self.leftTopCoordsOfBox(boxx, boxy)
        pygame.draw.rect(self.DISPLAYSURF, HIGHLIGHTCOLOR, (left - 5,
                                                            top - 5, self.BOXSIZE + 10, self.BOXSIZE + 10), 2)

    def drawHighlightWall(self, boxx, boxy, third_element):
        if third_element == 1:
            left = self.XMARGIN + (self.BOXSIZE + self.GAPSIZE) * boxx
            top = self.YMARGIN + (self.BOXSIZE + self.GAPSIZE) * \
                boxy + self.BOXSIZE
            pygame.draw.rect(self.DISPLAYSURF, HIGHLIGHTCOLOR, (left + 5,
                                                                top + 5, self.BOXSIZE + self.GAPSIZE + self.BOXSIZE - 10, self.GAPSIZE - 10), 0)
        if third_element == 2:
            left = self.XMARGIN + \
                (self.BOXSIZE + self.GAPSIZE) * boxx + self.BOXSIZE
            top = self.YMARGIN + (self.BOXSIZE + self.GAPSIZE) * boxy
            pygame.draw.rect(self.DISPLAYSURF, HIGHLIGHTCOLOR, (left + 5,
                                                                top + 5, self.GAPSIZE - 10, self.BOXSIZE + self.GAPSIZE + self.BOXSIZE - 10), 0)

    def gameWonAnimation(self):
        # flash the background color when the player has won
        color1 = LIGHTBGCOLOR
        color2 = BGCOLOR

        for i in range(10):
            color1, color2 = color2, color1  # swap
            self.DISPLAYSURF.fill(color1)
            pygame.display.update()
            pygame.time.wait(300)

    def updateLegalActionAndState(self):
        if self.player1_on and self.player2_on:
            if self.env.last_played == self.player1_num:  # agent2 가 다음 플레이어일 경우
                self.legal_action = self.env.get_legal_action(
                    self.env.get_state(self.player2_num))
            else:
                self.legal_action = self.env.get_legal_action(
                    self.env.get_state(self.player1_num))
        elif self.player1_on:
            self.legal_action = self.env.get_legal_action(
                self.env.get_state(self.player1_num))
        elif self.player2_on:
            self.legal_action = self.env.get_legal_action(
                env.get_state(self.player2_num))
        else:
            self.legal_action = None
        print("legal action:", self.legal_action)
        self.wall_map = self.env.map
        self.player_map = self.getPlayerMap(self.env)

    def checkObjLegalAndGetMove(self, boxx, boxy, third_element):
        width = self.width
        playerNo = -1  # 플레이 가능한 플레이어의 number 에 대한 변수
        if self.player1_on and self.env.last_played != self.player1_num:  # player1의 입장에서 Legal 여부 판단
            playerNo = 1
            x = boxx
            if third_element == 2 or third_element == 1:  # 벽설치의 경우 변환이 조금 다름.
                y = width - 2 - boxy
            else:
                y = width - 1 - boxy
        if self.player2_on and self.env.last_played != self.player2_num:  # player2의 입장에서 Legal 여부 판단
            playerNo = 2
            x = boxx
            if third_element == 2 or third_element == 1:  # 벽설치의 경우 변환이 조금 다름.
                y = width - 2 - boxy
            else:
                y = width - 1 - boxy

        # p1의 차례로 player1의 입장에서 Legal 여부 판단
        if playerNo == 1:
            if third_element == 0:  # 움직임 관련
                # 북진 가능 N
                if y > 0:
                    if self.player_map[x][y - 1] == playerNo or self.player_map[x][y - 1] == 3:
                        if ACT_MOVE_NORTH in self.legal_action:
                            self.input_action = [
                                self.player1_num, ACT_MOVE_NORTH]
                            return True
                # 서진 가능 W
                if x < width - 1:
                    if self.player_map[x + 1][y] == playerNo or self.player_map[x + 1][y] == 3:
                        if ACT_MOVE_WEST in self.legal_action:
                            self.input_action = [
                                self.player1_num, ACT_MOVE_WEST]
                            return True
                # 남진 가능 S
                if y < width - 1:
                    if self.player_map[x][y + 1] == playerNo or self.player_map[x][y + 1] == 3:
                        if ACT_MOVE_SOUTH in self.legal_action:
                            self.input_action = [
                                self.player1_num, ACT_MOVE_SOUTH]
                            return True
                # 동진 가능 E
                if x > 0:
                    if self.player_map[x - 1][y] == playerNo or self.player_map[x - 1][y] == 3:
                        if ACT_MOVE_EAST in self.legal_action:
                            self.input_action = [
                                self.player1_num, ACT_MOVE_EAST]
                            return True
            # 가로 벽
            elif third_element == 1:
                if x + (y) * (width - 1) + ACT_MOVE_CNT in self.legal_action:
                    self.input_action = [self.player1_num,
                                         x + (y) * (width - 1) + ACT_MOVE_CNT]
                    return True
            elif third_element == 2:
                if x + (width - 1) * (width - 1) + (y) * (width - 1) + ACT_MOVE_CNT in self.legal_action:
                    self.input_action = [
                        self.player1_num, x + (width - 1) * (width - 1) + (y) * (width - 1) + ACT_MOVE_CNT]
                    return True
        # player 2의 차례로 p2 관점에서 관찰
        elif playerNo == 2:
            if third_element == 0:  # 움직임 관련
                # 북진 가능 N
                if y < width - 1:
                    if self.player_map[x][y + 1] == playerNo or self.player_map[x][y + 1] == 3:
                        if ACT_MOVE_NORTH in self.legal_action:
                            self.input_action = [
                                self.player2_num, ACT_MOVE_NORTH]
                            return True
                # 서진 가능 W
                if x > 0:
                    if self.player_map[x - 1][y] == playerNo or self.player_map[x - 1][y] == 3:
                        if ACT_MOVE_WEST in self.legal_action:
                            self.input_action = [
                                self.player2_num, ACT_MOVE_WEST]
                            return True
                # 남진 가능 S
                if y > 0:
                    if self.player_map[x][y - 1] == playerNo or self.player_map[x][y - 1] == 3:
                        if ACT_MOVE_SOUTH in self.legal_action:
                            self.input_action = [
                                self.player2_num, ACT_MOVE_SOUTH]
                            return True
                # 동진 가능 E
                if x < width - 1:
                    if self.player_map[x + 1][y] == playerNo or self.player_map[x + 1][y] == 3:
                        if ACT_MOVE_EAST in self.legal_action:
                            self.input_action = [
                                self.player2_num, ACT_MOVE_EAST]
                            return True
            # 가로 벽
            elif third_element == 1:
                if (width - 2 - x) + (width - 2 - y) * (width - 1) + ACT_MOVE_CNT in self.legal_action:
                    self.input_action = [
                        self.player2_num, (width - 2 - x) + (width - 2 - y) * (width - 1) + ACT_MOVE_CNT]
                    return True
            elif third_element == 2:
                if (width - 2 - x) + (width - 1) * (width - 1) + (width - 2 - y) * (width - 1) + ACT_MOVE_CNT in self.legal_action:
                    self.input_action = [self.player2_num, (width - 2 - x) + (width - 1) * (
                        width - 1) + (width - 2 - y) * (width - 1) + ACT_MOVE_CNT]
                    return True
        return False

    def confirmMove(self):
        state, reward, done = self.env.step(
            self.input_action[0], self.input_action[1])
        if self.input_action[0] == self.player1_num:
            print("플레이어 1의 ", end="")
        if self.input_action[0] == self.player2_num:
            print("플레이어 2의 ", end="")
        print("action", self.input_action[1], "의 보상", reward)
        self.game_done = done


q = QuoridorEnv(width=5, value_mode=9)
agent_1 = q.register_agent()
agent_2 = q.register_agent()
dumb = DumbAgent(q, agent_2)
mcts = Agent(q, agent_2)
dumb2 = DumbAgent(q, agent_1)
print(q.get_legal_action(q.get_state(agent_1)))
q.render(agent_1)
# g = QuoridorGUI(q, agent_num1=agent_1, agent_num2=agent_2,
#                 auto_pilot1=None, auto_pilot2=None)
g = QuoridorGUI(q, agent_num1=agent_1,
                auto_pilot1=mcts, auto_pilot2=None)
# g = QuoridorGUI(q, auto_pilot1=dumb, auto_pilot2=dumb2)
g.startGame()
