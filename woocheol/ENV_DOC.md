# Env Doc

### Quridor(width, wall_cnt)

쿼리도 게임 객체를 생성하는 생성자입니다.

> width: 게임 판의 가로 세로크기를 지정합니다

> wall_cnt: 플레이어당 놓을 수 있는 벽의 개수입니다.

### Object.init_rendering()

렌더링을 할 경우 보여주는데 필요한 요소들을 불러오는 함수입니다. 뭐리도 객체를 불러오기 전에 한번만 호출해주면 됩니다. 학습할 경우 렌더링을 하지 않을때는 이용할 필요가 없습니다.

### Object.reset()

게임을 초기화 시킬때 이용됩니다.
벽의 상태 및 플레이어의 위치, 놓을 수 있는 벽의 개수가 초기화됩니다.

### Object.render()

게임을 렌더링합니다. 학습할 경우엔 보여줄 필요가 없습니다.

### Object.step(action)

게임을 진행합니다. 진행할 때 마다 플레이어가 바뀌게 됩니다.

> action: 플레이어의 액션입니다. action의 값은 다음과 같습니다.
>
> - 0 ~ 3 : 상하좌우로 이동합니다
> 
> - 4 ~ w*w + 3 : 가로로 벽을 세웁니다.
> 
> - w*w + 4 ~ 2*w*w + 3 : 세로로 벽을 세웁니다.

> return value: state, reward, done
> - state: 게임의 상태를 나타냅니다.
> [p1_turn, ( P1_x, P1_y, P1_wall_cnt ), ( P2_x, P2_y, P2_wall_cnt ), WallPos ]
> 
> (WallPos는 [w*2][w]로 구성된 배열입니다.)
> - reward: 보상입니다.
> - done: 끝났는지에 대한 여부입니다.