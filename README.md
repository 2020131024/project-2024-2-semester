Motivation:
야구선수에 대한 과학적이고 계랑적 평가를 위해 현대야구분석에서는 '세이버메트릭스'방법을 사용한다.
고전적인 야구선수에 대한 평가방식인 '타율'은 현대야구분석에서는 크게 중요하지 않은 요소로 평가받는다.
선수를 분석할 때 '타율'말고 계랑적인 방법으로 'OPS', 'WAR', 'WRC+'가 주로 사용되며 이 3요소는 귀납적으로
최근 현대야구에서 좋은 선수를 나타내는 지표가 되었다. '타율'과 다른 3지표의 상관관계를 파악하고자 한다.

DATA 획득:
한국프로야구의 선수별 데이터를 잘 정리한 스탯티즈 사이트에서 2023년도 타율상위20명의 DATA를 얻은 후 엑셀로 정리
DATA파일은 data3.csv파일로 올라가있다.
https://statiz.sporki.com/special/?m=main

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


