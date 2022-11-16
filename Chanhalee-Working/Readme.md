# chanhalee 의 작업 브랜치

started: 2022.11.04

## ENV

### 동작 개요

모든 에이전트는 자신의 학습에서 자신을 p1으로 인식.<br>
__BUT!__ 2개의 에이전트가 하나의 ENV를 공유하게되어서 ENV에서는 자신을 p1으로 여기는 agent들을 구분할 방법이 있어야한다.

이를 위해 에이전트는 ENV에 자신을 등록하는 절차를 거쳐 고유번호를 부여받고, 이를 보관한다.<br>
ENV의 각종 메서드를 사용할때 고유번호를 env에 함께 넘기고, ENV에서는 이를 이용해 이 에이전트가 어떤 에이전트인지 식별하게된다.

에이전트를 등록하는 절차는 QuoridorEnv.register_agent()를 통해 가능하며 반환값으로 에이전트는 agent_num을 받는다.<br>
반환값은 int값이며 agent에 잘 보관해야한다.

### state

QuoridorEnv.get_state(agent_num) 을 통해 획득 가능.

(map, playerState) 으로 구성된 튜플이 반환됨.<br>
map 과 playerState는 numpy.ndarray이다.

* map

	map[2][width - 1][width - 1] 크기의 boolean 3차원 배열.<br>
	map[0][][]는 가로벽,<br>
	map[1][][]는 세로벽

* playerState

	playerState[2][3] 형식의 int 2차원 배열<br>
	playerState[0] 은 나의 위치를 보관<br>
	playerState[1] 은 상대의 위치를 보관<br>
	playerState[][0]: position_x<br>
	playerState[][1]: position_y<br>
	playerState[][2]: wall_left

### action 

최초 약속대로 4 + (width - 1) * (width - 1) * 2 가지의 명령이 가능하다.

즉, width = 5일 경우 0 ~ 35 범위의 action이 가능

* 0 ~ 3

	NWSE 이동

* 3 ~ 3 + (width - 1) * (width - 1)

	가로로 벽을 설치

* 3 + (width - 1) * (width - 1) ~ 3 + (width - 1) * (width - 1) * 2

	세로로 벽을 설치

* 참고 그림

	![img](/Chanhalee-Working/etc/action_example.jpeg)

### legal_action

아마 많은 에이전트에서 필요로 할 것으로 보여 env 함수에 구현해두었다.
입력값으로 state를 주면된다. state: (map, player_status)
state 는 get_state 함수를 이용해 확인하길 바람. numpy array 2개로 이루어진 튜플입니다.

### QuoridorEnv 초기화

QuoridorEnv 의 초기화에 사용된다.<br>
**__init__(width=5, value_mode=0)**

* width 

	게임판의 가로 세로 길이에 대한 정보이다.<br>
	반드시 4 초과 10 미만의 홀수가 입력되어야 하며 위반시 예외가 발생한다.

* value_mode

	state 의 value를 어떻게 평가할지에 대한 함수이다.<br>
	0~5 사이의 값을 가져야 한다.<br>
	각 모드에 대한 설명은 아래 get_value(agent_num) 항목에서 하겠다.

### 함수 목록

* register_agent()
* reset(width=-1, value_mode=-1)
* get_legal_action(state)
* render(agent_num)
* step(agent_num, action)
* get_state(agent_num)
* get_flipped_state(state)
* ask_how_far(state)
* ask_how_far_opp(state)
* get_value(agent_num)
* ask_end_state(state)

#### register_agent()

에이전트를 등록하는 메서드이다.<br>
하나의 ENV에는 최대 2개의 에이전트가 등록될 수 있으며 3개 이상을 등록하려 할 경우 Exception이 발생한다.

return: int

#### reset(width=-1, value_mode=-1)

__\_\_init\_\___ 을 다시 호출하여 QuoridorEnv를 다시 초기화 하는 메서드이다.<br>
width와 value_mode 를 따로 입력하지 않을 경우 기존의 width와 value_mode 를 유지한다.

return: none

#### get_legal_action(state)

__util function__ 으로 agent 구현의 편의를 위해 구현한 함수이다.

state에 p1의 위치에 있는 플레이어의 legal action을 반환하는 메서드다.<br>
이용할 때 get_state(agent_num) 메서드의 반환값을 주는 것을 권장한다.<br>
연산할거리가 조금 많아서 남용할 경우 실행속도 저하를 불러온다.<br>
반환값은 가능한 엑션들의 번호가 담긴 튜플이다.

return: (int, int, int, ...)

#### render(agent_num)

콘솔 창에 agent_num 이 p1으로 보이도록 출력해주는 함수이다.

return: none

#### step(agent_num, action)

step을 진행하는 함수이다.<br>
에이전트의 action을 받아서 env 내부의 state를 변화시키고,<br>
action의 결과로 도착한 state와 해당 state의 value를 반환한다.<br>
또한 해당 action의 결과로 게임이 끝났는지 여부도 반환한다.

__반드시!! legal action만 입력되어야 한다!__

return: state, step_reward, step_done<br>
```
state: (map, player_state)
step_reward: int
step_done: boolean
```

#### get_state(agent_num)

agent_num을 갖는 플레이어가 p1으로 세팅된 state를 반환한다.

return: state
```
state: (map, player_state)
```

#### get_flipped_state(state)

__util function__ 으로 사용을 권장하지 않으며 에이전트가 사용할 일은 아마 없을 것이다.<br>
state상 p2이 p1이 되도록 뒤집은 copy를 반환한다.

return: state
```
state: (map, player_state)
```

#### ask_how_far(state)

__util function__ 으로 state의 p1이 종료상태까지 도달하기 위해 최소 몇 수가 소요되는지 반환한다.<br>
state 상 배치된 벽을 고려하기에 연산이 약간 있다.<br>
반환값으로 -1이 주어질 수 있는데 이는 어떠한 방법으로도 목표 상태에 도달할 수 없는 경우이다.

return: int
```
 -1: error state
  0: end state
etc: how many moves
```

#### ask_how_far_opp(state)

__util function__ 으로 p2가 종료상태까지 남은 수를 반환한다.<br>
규칙은 ask_how_far(state)와 동일하다.

return: int

#### get_value(agent_num)

__util function__ 으로 MDP를 모르는 가정이 있기에 임의의 state에 대해 value를 계산해주는 기능은 넣지 않았다.<br>
agent_num 에게 ENV가 보관중인 상태의 가치가 얼마인지 알려준다.<br>
직전 step의 결과와 같은 값을 알려주니 사용하지 않을 것으로 생각한다.

**value functions**

내부에 여러가지 value function이 구현되어 있으며 이는 QuoridorEnv 초기화시 어떤 value function을 사용할지 정할 수 있다.

* 0: 가장 간단한 value_function

    승리시 150<br>
    패배시 -150<br>
    그외 -1

* 1: 약간 복잡한 value_function

	승리시 100<br>
	패배시 -100<br>
	그외 y축 값에 따라 차등지급<br>
	산식: reward = ($도착 라인과 거리 (벽무시)) * -1<br>
	ex) 5 x 5 게임판에서 (1, 2): -2, (3, 4): 200, (4, 3): -1

* 2: 조금 더 복잡한 value_function

	__주의점__: nq learning 할때 깊이를 충분히 깊게 탐색해야할 것. (아예 end state까지 탐색을 권장)<br>
	이 value_function을 도입한 에이전트는 티배깅(인성질)을 할 가능성이 있음...<br>
	승리시 1000<br>
	패배시 -1000<br>
	그 외 "상대와 나의" y축 값에 따라 차등지급<br>
	산식: reward = (상대의 end_line 과의 거리 (벽무시)) - (나의 end_line 과의 거리 (벽무시)) * 2 - 1

* 3: 많이 복잡한 value_function
	
	__주의점__: 연산량이 상당하기에 탐색 범위가 넓으면 학습에 상당한 시간이 걸릴 것임.<br>
	장점: 직관적으로 고개가 끄덕여지는 reward 를 반환.<br>
    승리시 1000<br>
    패배시 -1000<br>
	그 외  {승리 조건까지 도달하기에 얼마나 남았는지 벽을 포함하여 연산한 값} * -1

* 4: 아주 많이 복잡한 value_function
    
	__주의점__: 연산량이 상당하기에 탐색 범위가 넓으면 학습에 상당한 시간이 걸릴 것임.<br>
    장점: 직관적으로 고개가 끄덕여지는 reward 를 반환.<br>
    승리시 1000<br>
    패배시 -1000<br>
	그 외 {상대의 도착까지 남은 수} - {승리 조건까지 도달하기에 얼마나 남았는지 벽을 포함하여 연산한 값} * 2 -1

return: int

#### ask_end_state(state)

__util function__ 으로 해당 상태가 종료상태인지 확인해준다.<br>
종료상태일 경우 어떤 agent가 승리했는지 agent_num을 반환한다.

return: int
```
0: not end state
AGENT_1: agent_num == AGENT_1 have won
AGENT_2: agent_num == AGENT_2 have won
```

