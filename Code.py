
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
train=pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Multi Class Obesity Risk\train.csv")
test=pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Multi Class Obesity Risk\test.csv")
print(train.info())
# # drop colums 
# X_train=train.drop(columns=['id','NObeyesdad'])
# Y_train=train['NObeyesdad']
# Test=test.drop(columns=['id'])

# nc=X_train.select_dtypes(include=np.number).columns
# cc=X_train.select_dtypes(exclude=np.number).columns

# X_train=pd.get_dummies(X_train,columns=cc,drop_first=True)
# Test=pd.get_dummies(Test,columns=cc,drop_first=True)

# Test=Test.reindex(columns=X_train.columns,fill_value=0)

# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# Test=sc.fit_transform(Test)
# L=LabelEncoder()
# Y_train=L.fit_transform(Y_train)

# X_tr,X_val,Y_tr,Y_val=train_test_split(X_train,Y_train,test_size=0.05,random_state=42)

# ml=XGBClassifier(objective="multi:softmax",n_estimators=100,learning_rate=0.15,device='cuda:0',eval_metric='merror')
# ml.fit(X_tr,Y_tr)
# pred_val=ml.predict(X_val)
# acc=accuracy_score(pred_val,Y_val)
# cr=classification_report(pred_val,Y_val)
# # print(f"acc={acc}")
# # print(f"cr:.{cr}")

# cv=cross_val_score(ml,X_train,Y_train,cv=20,scoring='accuracy')
# print(f"cv mean{cv.mean()}")
# pred_numeric=ml.predict(Test)


# pred=L.inverse_transform(pred_numeric)

# sub=pd.DataFrame({
#     'id':test['id'],'NObeyesdad':pred
# })

# sub.to_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Playgraoung\Multi Class Obesity Risk\sample_submission.csv",index=False)
