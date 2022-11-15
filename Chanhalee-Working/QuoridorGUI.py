import random
import pygame
import sys
from pygame.locals import *

FPS = 30  # frames per second, the general speed of the program

REVEALSPEED = 8  # speed boxes' sliding reveals and covers

#            R    G    B
GRAY = (100, 100, 100)
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
    def __init__(self, env):
        pygame.init()
        self.WINDOWWIDTH = 640  # size of window's width in pixels
        self.WINDOWHEIGHT = 480  # size of windows' height in pixels
        self.BOXSIZE = 40  # size of box height & width in pixels
        self.GAPSIZE = 15  # size of gap between boxes in pixels
        self.width = env.width
        self.env = env
        self.FPSCLOCK = pygame.time.Clock()
        self.DISPLAYSURF = pygame.display.set_mode(
            (self.WINDOWWIDTH, self.WINDOWHEIGHT))
        self.XMARGIN = int(
            (self.WINDOWWIDTH - (self.width * (self.BOXSIZE + self.GAPSIZE))) / 2)
        self.YMARGIN = int(
            (self.WINDOWHEIGHT - (self.width * (self.BOXSIZE + self.GAPSIZE))) / 2)

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

    def startGame(self):
        while True:  # main game loop
            self.mouseUp = False
            self.DISPLAYSURF.fill(BGCOLOR)  # drawing the window
            self.drawBoard()

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
                    if not self.mouseDown:
                        self.oldMousex = self.mousex
                        self.oldMousey = self.mousey
                    self.mousex, self.mousey = event.pos
                    self.mouseUp = True
                    self.mouseDown = False
            boxx, boxy, third_element = self.getObjAtPixel(
                self.oldMousex, self.oldMousey, self.mousex, self.mousey)
            if third_element != -1:
                # The mouse is currently over a box.
                if third_element == 1 or third_element == 2:
                    self.drawHighlightWall(boxx, boxy, third_element)
                    self.drawHighlightBox(boxx, boxy)
                if third_element == 0:
                    self.drawHighlightBox(boxx, boxy)

                    # Redraw the screen and wait a clock tick.
            pygame.display.update()
            self.FPSCLOCK.tick(FPS)

    def generateStartingPos(self, val):
        player_map = []
        for i in range(self.width):
            player_map.append([val] * self.width)
        return player_map

    def getRandomizedBoard():
        # Get a list of every possible shape in every possible color.
        icons = []
        for color in ALLCOLORS:
            for shape in ALLSHAPES:
                icons.append((shape, color))

        random.shuffle(icons)  # randomize the order of the icons list
        # calculate how many icons are needed
        numIconsUsed = int(self.width * self.width / 2)
        icons = icons[:numIconsUsed] * 2  # make two of each
        random.shuffle(icons)

        # Create the board data structure, with randomly placed icons.
        board = []
        for x in range(self.width):
            column = []
            for y in range(self.width):
                column.append(icons[0])
                del icons[0]  # remove the icons as we assign them
            board.append(column)
        return board

    def splitIntoGroupsOf(groupSize, theList):
        # splits a list into a list of lists, where the inner lists have at
        # most groupSize number of items.
        result = []
        for i in range(0, len(theList), groupSize):
            result.append(theList[i:i + groupSize])
        return result

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
        top = boxy* (self.BOXSIZE +
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
        print(x_a, y_a)
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

    def getShapeAndColor(self, board, boxx, boxy):
        # shape value for x, y spot is stored in board[x][y][0]
        # color value for x, y spot is stored in board[x][y][1]
        return board[boxx][boxy][0], board[boxx][boxy][1]

    def drawBoard(self):
        # player가 설 수 있는 위치들을 그림
        for boxx in range(self.width):
            for boxy in range(self.width):
                left, top = self.leftTopCoordsOfBoxFlipped(boxx, boxy)
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
                        (self.width - boxy - 1) + self.BOXSIZE
                    pygame.draw.rect(self.DISPLAYSURF, ORANGE, (left + 5,
                                                                top + 5, self.BOXSIZE + self.GAPSIZE + self.BOXSIZE - 10, self.GAPSIZE - 10), 0)
                if self.wall_map[1][boxx][boxy]:
                    left = self.XMARGIN + \
                        (self.BOXSIZE + self.GAPSIZE) * boxx + self.BOXSIZE
                    top = self.YMARGIN + \
                        (self.BOXSIZE + self.GAPSIZE) * (self.width - boxy - 1)
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

    def startGameAnimation(self, board):
        # Randomly reveal the boxes 8 at a time.
        coveredBoxes = generateRevealedBoxesData(False)
        boxes = []
        for x in range(self.width):
            for y in range(self.width):
                boxes.append((x, y))
        random.shuffle(boxes)
        boxGroups = splitIntoGroupsOf(8, boxes)

        drawBoard(board, coveredBoxes)
        for boxGroup in boxGroups:
            revealBoxesAnimation(board, boxGroup)
            coverBoxesAnimation(board, boxGroup)

    def gameWonAnimation(self, board):
        # flash the background color when the player has won
        coveredBoxes = generateRevealedBoxesData(True)
        color1 = LIGHTBGCOLOR
        color2 = BGCOLOR

        for i in range(13):
            color1, color2 = color2, color1  # swap colors
            self.DISPLAYSURF.fill(color1)
            drawBoard(board, coveredBoxes)
            pygame.display.update()
            pygame.time.wait(300)
