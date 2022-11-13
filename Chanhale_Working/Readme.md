## chanhalee 의 작업 브랜치

started: 2022.11.04

### ENV

#### 약속

모든 에이전트는 자신을 p1으로 인식.
에이전트의 초기화시 env를 사용하여 자신의 정보를 초기화
agent 초기화 방법: env.agent1, env.agent2 를 확인.
두 값은 boolean 값으로, 이미 차지된 자리는 true 로 표기되어 있음.
즉, false 로 표기된 자리가 비었다는 뜻.
false 로 표기된 값을 true로 세팅하고, 에이전트에 자신이 몇번째 에이전트 자리를 차지했는지 int값으로 보관

env 를 사용하는 거의 모든 함수의 첫번째 인자는 자신이 몇번째 에이전트인지에 대한 정보가 들어와야 함.
아래는 그 목록들임
step<br>
get_state<br>

#### legal_action

아마 많은 에이전트에서 필요로 할 것으로 보여 env 함수에 구현해두었습니다.
입력값으로 state를 주면된다. state: (map, player_status)
state 는 get_state 함수를 이용해 확인하길 바람. numpy array 2개로 이루어진 튜플입니다.

#### state

QuoridorEnv.get_state(agent_num) 을 통해 획득 가능.

(map, playerState) 으로 구성된 튜플이 반환됨.

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

#### action 

최초 약속대로 4 + (width - 1) * (width - 1) * 2 개수의 명령임.

즉, width = 5일 경우 0 ~ 35 범위의 action이 가능

* 0 ~ 3

	NWSE 이동

* 3 ~ 3 + (width - 1) * (width - 1)

	가로로 벽을 설치

* 3 + (width - 1) * (width - 1) ~ 3 + (width - 1) * (width - 1) * 2

	세로로 벽을 설치

* 참고 그림

	![img](/Chanhale_Working/etc/action_example.jpeg)