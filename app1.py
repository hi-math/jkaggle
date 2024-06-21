import streamlit as st
import pandas as pd
import time
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.sql import text
import datetime

st.set_page_config(layout="wide")

st.image("img/bg.png")

col1, col2 = st.columns([3, 1])

col1.title("무는 원숭이 분류하기")
col1.write("동물원으로부터의 의뢰입니다. 포악한 원숭이가 조련사도 물고 있습니다. 숙련된 조련사는 괜찮겠지만 초보들에게 원숭이는 위협적인 존재가 될 수 있습니다. 다행히도 무는 원숭이는 인상착의로 구분할 수 있다고 하는 군요. 여러분의 도움이 필요합니다.")

tab1, tab2, tab3, tab4 = st.tabs(["살펴보기", "데이터", "코드 설명", "순위"])

with tab1:
    st.header("개요")
    st.caption("이 대회의 목표는 원숭이의 이미지를 이용하여 무는 원숭이와 물지 않는 원숭이를 구분하는 알고리즘과 모델을 만드는 것입니다. 귀하의 참여는 동물원에 큰 도움이 되며 이미지 분류 모델 발전에 도움이 됩니다.")
    st.write("")
    st.divider()
    st.header("설명")
    st.caption("자료의 속성을 파악하는 것은 인공지능의 중요한 기본입니다. 수학적 추론은 과학에서뿐 아니라 금융이나 공학 등 다양한 영역에서 복잡한 문제를 해결하는 훌륭한 수단이 됩니다.")
    st.caption("기계학습을 이용하여 이미지의 속성을 파악하고 이를 분류하는 실습을 통해 인공지능이 데이터를 어떻게 다루는지 학습하고 실제 환경과 비슷한 체험을 할 수 있습니다.")
    st.caption("이 과제를 해결하기위해 우리는 몇몇 원숭이의 인상착의를 준비했습니다. 데이터의 속성을 파악하고 정확도 높은 원숭이 분류 모델을 생성해보세요.")
    st.write("")
    st.divider()
    st.header("평가")
    st.caption("제출물은 예측 라벨과 실제 라벨의 정확성을 기준으로 평가합니다. 즉, 제출물은 실제 라벨과 정확히 일치하는 예측 라벨의 비율에 따라 순위가 매겨집니다.")
    st.caption("이 대회에서 정확성은 100%를 기준으로 측정되며 소숫점 둘째 자리에서 반올림하여 첫째자리까지 표시됩니다.")
    st.caption("단 모델의 과적합을 방지하기 위해서 3마리의 원숭이는 비공개 데이터로 남겨둡니다. 상위 5팀에 선정되면 비공개 데이터를 분류하는 능력을 포함시켜 순위가 결정됩니다.")
    st.write("")
    st.divider()
    st.header("제출")
    st.caption("모델을 통해 예측된 값을 제출 탭에서 :blue-background[submission.csv] 파일로 업로드하세요.")

with tab2:
    col1, col2 = st.columns([3,1])    
    col1.header("데이터 셋 설명")
    with open('datasets.zip', 'rb') as f:
        col2.download_button('Download Datasets', f, file_name='datasets.zip')
    
    st.caption("이 대회에서 사용하는 데이터는 훈련데이터와 테스트 데이터가 있습니다.")
    st.caption("훈련데이터는 모델 학습에 사용되는 데이터입니다. 이미지와 함께 라벨이 제공됩니다.")
    st.caption("테스트 데이터는 모델의 성능을 확인하기 위한 데이터로 라벨이 제공되지 않습니다.")
    st.write("")
    st.divider()
    st.header("파일")
    st.caption("train.csv - 훈련 데이터로 사용할 수 있는 문제의 라벨이 포함되어 있습니다.")
    st.caption("test.csv - 10개의 문제가 포함되어 있습니다.")
    st.caption("Sample_submission.csv - 올바른 형식의 샘플 제출 파일입니다. 제출 형식에 대한 자세한 내용은 평가 페이지를 참조하세요.")
    
with tab3:
    st.header("코드 설명")
    st.caption("데이터를 다루기 위한 판다스와 의사결정나무 분류기를 불러옵니다.")
    code = '''
    import pandas as pd 
    from sklearn.tree import DecisionTreeClassifier
    '''
    st.code(code, language='python')
    st.divider()
    
    st.caption("train.csv를 불러옵니다.")
    code = '''
    data = pd.read_csv('train.csv')
    '''
    st.code(code, language='python')
    st.divider()
    
    st.caption("학습데이터와 정답을 나눕니다. 학습데이터는  id와 정답을 삭제하여 우리가 입력한 속성값만 가집니다. 정답데이터는 표에서 LABEL 열만 데이터로 남깁니다.")
    code = '''
    X_train = data.drop(['ID', 'label'], axis=1)
    y_train = data['label']
    '''
    st.code(code, language='python')
    st.divider()
    
    st.caption("모델을 학습시킵니다. 이때 속성데이터와 정답데이터를 가지고 학습합니다.")
    code = '''
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    '''
    st.code(code, language='python')
    st.divider()

    st.caption("모델이 어떻게 학습되었는지 확인합니다.")
    code = '''
    import matplotlib.pyplot as plt
    from sklearn import tree

    plt.figure( figsize=(20,15) )
    tree.plot_tree(model, 
               class_names=['0','1'],
               feature_names=X_train.columns,
               impurity=True, filled=True,
               rounded=True)
    '''
    st.code(code, language='python')
    st.divider()
    
    st.caption("이제 우리가 만든 모델로 예측하기 위해서 테스트 데이터를 불러옵니다. 그리고 테스트 데이터의 속성만 불리해봅시다.")
    code = '''
    X_test = test_data.drop(['ID', 'label'], axis=1)
    y_test = test_data['label']
    '''
    st.code(code, language='python')
    st.divider()
    
    st.caption("테스트 데이터의 속성을 이용하여 결과를 예측해봅니다.")
    code = '''
    predict = model.predict(X_test)
    '''
    st.code(code, language='python')
    st.divider()
    
    st.caption("예측결과를 저장하고 파일로 다운로드 받아봅시다.")
    code = '''
    test_data['label'] = predict
    submission = test_data[['ID', 'label']]
    pd.DataFrame(submission).to_csv('submission.csv', index=False)
    '''
    st.code(code, language='python')
    st.divider()


with tab4:
    login = {"권세빈":1104,
             "전진":1120,
             "박연수":1208,
             "송이랑":1210,
             "김정현":1404,
             "양지우":1412,
             "권보솔":1602,
             "김서희":1603,
             "김윤서":1608,
             "변서후":1613,
             "박윤수":1808,
             "안수완":1810,
             "윤이든":1813,
             "하강우":1919,
             "윤서정":2317,
             "허예솔":2322,
             "권오윤":2402,
             "황인영":2425,
             "김윤":2906,
             "오태규":3815
            }
    connection_string = "mysql+pymysql://user:password@host:port/dbname"
    conn = st.connection('mysql', type='sql')   
    col1, col2 = st.columns([3,1])
    col1.header("리더보드")
    
    if st.button("새로고침",key="re"):
        conn.reset()
        conn = st.connection('mysql', type='sql')   
        
    df = conn.query('SELECT * FROM mon;', ttl=600)
    def aggregate_group(group):
        # 값이 가장 큰 행
        largest_value_row = group.loc[group['score'].idxmax()]
        
        # 합산된 시도횟수
        total_attempts = group['try'].sum()
        
        return pd.Series({
            'name': largest_value_row['name'],
            'score': largest_value_row['score'],
            'try': total_attempts,
            'datetime': largest_value_row['datetime']
        })

    result = df.groupby('name', group_keys=False).apply(lambda group: aggregate_group(group)).reset_index(drop=True)

    result = result[['name', 'score', 'try', 'datetime']]
    result = result.rename(columns={'name': '이름', 'score': '정확도', 'try': '시도횟수', 'datetime':'datetime'})
    result = result.sort_values(by=['정확도', 'datetime'], ascending=[False, True])
    result
 

    @st.experimental_dialog("upload")
    def score():
        name = st.text_input("이름")
        uploaded_files = st.file_uploader("Choose a CSV file", type=['csv'], accept_multiple_files=False)        
        if uploaded_files is not None:
            try:
                file = pd.read_csv(uploaded_files)
                pred = list(file['label'])
                for i in pred:
                    if i==0 or i==1 and len(pred)==10:
                        isfile = 0
                    else:
                        isfile = 1
                        st.warning("0과 1만 입력할 수 있습니다.")
                        break
                sub = False
                if isfile==0:
                    file
                    sub = st.button("제출", key='a1')     
                       
                if sub:
                    now = datetime.datetime.now()                    
                    def sco(list)->float:
                        answer = [0,1,0,0,1,1,0,0,0,0]
                        right = 0
                        for i in range(len(answer)):
                            if answer[i] == pred[i]:
                                right+=1
                        acc = right/len(answer)*100
                        return round(acc,2)
                    
                    score = sco(pred)
                    trynn = 1
                    st.write(f"이번 시도의 점수는 {score}입니다.")                    
                    with conn.session as s:
                        query = text(f'INSERT INTO mon VALUES ("{now}", "{name}", {score},{trynn});')
                        s.execute(query)
                        s.commit()
                        button = False
                        s.close()
                
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a CSV file.")
        
    if col2.button("제출하기"):
        score()
     
