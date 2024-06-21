Motivation:
야구선수에 대한 과학적이고 계랑적 평가를 위해 현대야구분석에서는 '세이버메트릭스'방법을 사용한다.
고전적인 야구선수에 대한 평가방식인 '타율'은 현대야구분석에서는 크게 중요하지 않은 요소로 평가받는다.
선수를 분석할 때 '타율'말고 계랑적인 방법으로 'OPS', 'WAR', 'WRC+'가 주로 사용되며 이 3요소는 귀납적으로
최근 현대야구에서 좋은 선수를 나타내는 지표가 되었다. '타율'과 다른 3지표의 상관관계를 파악하고자 한다.

타율: 안타/타수
OPS: 출루율+장타율
WAR: Wins Above Replacement(대체 수준 대비 승리 기여도)
WRC+:Weighted Runs Created(조정 득점 창출력) + 각 구장별 특성(파크팩터) (구장별 바람, 펜스길이 등이 다르므로 이 지표를 반영한다.)

DATA 획득:
한국프로야구의 선수별 데이터를 잘 정리한 스탯티즈 사이트에서 2023년도 타율상위20명의 DATA를 얻은 후 엑셀로 정리
DATA: 타율, OPS, WAR, WRC+. DATA파일은 data3.csv파일로 올라가있다.
다만 data3.csv파일을 다운로드하면 한글이 깨지므로 data3.xlsx파일을 같이 올려두었다.
data3.xlsx파일을 파일형식-csv, 도구-웹 옵션-인코딩-문서를 다음 형식으로 저장-유니코드(UTF-8)로 저장한 후 사용하면 된다.
출처 : https://statiz.sporki.com/special/?m=main

MODEL:
pandas, numpy, matplotlib 모델을 사용하였다.

Performance:
이번 프로젝트에서는 2023년 한국프로야구 타율상위 20명을 기준으로 각 선수기준 '타율', 'OPS', 'WAR', 'WRC+'을
pandas, numpy, matplotlib를 통해 그래프로 나타냄으로써 시각적으로 어느정도의 유사성을 보이는 지 확인한다.
그래프에 한글이 들어가므로 나눔고딕폰트를 matplotlib 폰트에 추가하였으며 파일은 NanumGothic.otf로 파일을 올렸다.

Project1에서는 '타율', 'OPS', 'WAR'을 그래프로 나타냄으로써 3지표의 상관관계를 파악한다.
이 때 'WAR'이 그래프에 잘 드러나도록 'WAR'지표대신 '0.1WAR'지표를 사용하였다.
그래프는 Project1.타율,OPS,WAR 사진으로 올라가있다.

Project2에서는 '타율', 'OPS', 'WRC+'를 그래프로 나타냄으로써 3지표의 상관관계를 파악한다.
이 때 'WRC+'가 지표에 잘 드러나도록 'WRC+'지표대신 '0.003WAR'지표를 사용하였다.
그래프는 Project2.타율,OPS,WRC+ 사진으로 올라가있다.

수행결과:
Project1: 타율차이가 크지 않은 20명선수의 OPS를 타율과 비교해보면 타율이 높다고 OPS가 반드시 높지는 않다는 것을 알 수 있다.
1~13등은 전반적으로 조금씩 낮아지는 경향이 있지만 타율 15, 16등이 OPS 1, 2등을 차지한 것을 보면 타율과 OPS는 어느정도는
연관성이 있지만 실제 OPS등수를 반영하기에는 부족하다는 것을 알 수있다.
OPS와 WAR은 그래프의 고저가 차이가 많이 나지만 전반적으로 비슷한 계형을 가지는 것을 확인할 수 있다.

Project2: Project2에서는 OPS와 WRC+의 계형이 매우 흡사하다는 것을 확인할 수 있다.
OPS와 WRC+ 모두 현대야구분석에서 좋은 선수를 나타내는 기준임을 확인할 수 있다.
실제로 현대야구분석에서는 WRC+를 제일 중요한 지표중 하나로 보며, 한국프로야구는 WRC+ 지표를
거의 사용하지 않지만, 미국 MLB에서는 WRC+를 몇 년전부터 꾸준하게 중요한 지표로 활용하고 있다.

