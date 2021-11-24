import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################

# Creating a Header.

st.title("'merurafi' Streamlit App mini project")
st.header("Aplikasi Sweetviz dan Sckit")
st.markdown('Ini adalah penggunaan aplikasi tersebut untuk tujuan analisis data.')
st.subheader('Data set:')
st.subheader('"Rekod kes jenayah oleh remaja bagi tahun 2019 mengikut usia dan jantina."')
st.markdown('_Saya jumpa data berikut di laman web, cuba untuk membuat analisa mengenainya_.')


file =  pd.read_csv('2019_gender_kanak-kanak_jenayah_dan_usia.csv')

df = file.drop(['_id'], axis = 1)

# Install Sweetviz

import pandas as pd

file =  pd.read_csv('2019_gender_kanak-kanak_jenayah_dan_usia.csv')

df = file.drop(['_id'], axis = 1)

col_names = ['_id','Jenis Kesalahan','Umur 10 Hingga 12 Tahun Lelaki', 'Umur 10 Hingga 12 Tahun Perempuan', 'Umur 13 Hingga 15 Tahun Lelaki', 'Umur 13 Hingga 15 Tahun Perempuan','Umur 16 Hingga 17 Tahun Lelaki','Umur 16 Hingga 17 Tahun Perempuan','Umur 18 Hingga 21 Tahun Lelaki','Umur 18 Hingga 21 Tahun Perempuan'] 
df1 = pd.read_csv("2019_gender_kanak-kanak_jenayah_dan_usia.csv", header=0, names=col_names).drop(['_id'], axis = 1)
# df.head()

# importing sweetviz
import sweetviz as sv

#analyzing the dataset
analisa_laporan = sv.analyze(df)


#display the report
analisa_laporan.show_html('kanak_kanak_jenayah_dan_usia.html', open_browser=False)

import IPython
IPython.display.HTML('kanak_kanak_jenayah_dan_usia.html')

analisa_kiraan_laporan = sv.compare(df1[3:], df1[:2])

analisa_kiraan_laporan .show_html('kanak-kanak_jenayah_dan_usia.html')

df1['code'] = pd.factorize(df1['Jenis Kesalahan'])[0] + 1

from sklearn.model_selection import train_test_split

feature_cols = (['Umur 10 Hingga 12 Tahun Lelaki', 'Umur 10 Hingga 12 Tahun Perempuan', 'Umur 13 Hingga 15 Tahun Lelaki', 'Umur 13 Hingga 15 Tahun Perempuan','Umur 16 Hingga 17 Tahun Lelaki','Umur 16 Hingga 17 Tahun Perempuan','Umur 18 Hingga 21 Tahun Lelaki','Umur 18 Hingga 21 Tahun Perempuan'])

X = df1[feature_cols] # Features 
y = df1.code # Target variable

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

import sweetviz as sensivity
banding_laporan = sv.compare([X_train, 'Train'], [X_test, 'Test'])
banding_laporan.show_html('bandingan.html', open_browser=False)

# Design bar charts

# Boy chart
boy_selection = df1[['Jenis Kesalahan','Umur 10 Hingga 12 Tahun Lelaki','Umur 13 Hingga 15 Tahun Lelaki','Umur 16 Hingga 17 Tahun Lelaki','Umur 18 Hingga 21 Tahun Lelaki',]]
boy_selection_cols = ['kesalahan','Umur 10-12 Tahun ','Umur 13-15 Tahun','Umur 16-17 Tahun','Umur 18-21 tahun']
boy_selection.columns = boy_selection_cols 

# Girl chart
girl_selection = df1[['Jenis Kesalahan','Umur 10 Hingga 12 Tahun Perempuan','Umur 13 Hingga 15 Tahun Perempuan','Umur 16 Hingga 17 Tahun Perempuan','Umur 18 Hingga 21 Tahun Perempuan',]]
girl_selection_cols = ['kesalahan','Umur 10-12 Tahun ','Umur 13-15 Tahun','Umur 16-17 Tahun','Umur 18-21 tahun']
girl_selection.columns = boy_selection_cols 


df1_cols =['kesalahan','aL','aP','bL','bP','cL','cP','dL','dP','code']
df1.columns = df1_cols 

sum_1 = df1.iloc[0:10,1:3].sum(axis=1)
sum_2 = df1.iloc[0:10,3:5].sum(axis=1)
sum_3 = df1.iloc[0:10,5:7].sum(axis=1)
sum_4 = df1.iloc[0:10,7:9].sum(axis=1)

df2 = df1.drop(['aP','bP','cP','dP','code'], axis = 1)

df2['aL'].replace([55,36,5,0,64,0,1,29,1,27],[55,35,6,0,64,0,1,31,1,31],inplace=True)
df2['bL'].replace([414,143,16,0,255,4,5,93,3,103],[428,149,17,0,291,4,5,97,3,112],inplace=True)
df2['cL'].replace([1008,532,56,4,1223,31,24,208,9,423],[1051,542,58,4,1316,36,24,219,9,457],inplace=True)
df2['dL'].replace([70,51,3,1,155,5,3,8,0,78],[71,51,3,1,166,5,3,10,0,86],inplace=True)

df2_cols =['kesalahan','Umur 10-12 Tahun','Umur 13-15 Tahun','Umur 16-17 Tahun','Umur 18-21 tahun']
df2.columns = df2_cols 

both_gender_selection = df2

boy_selection['sums_boy'] = boy_selection.sum(axis=1)
boy_selection.iloc[:,1:5].sum(axis=1)
girl_selection['sums_girl'] = girl_selection.sum(axis=1)
girl_selection.iloc[:,1:5].sum(axis=1)
both_gender_selection['sums_both_gender'] = both_gender_selection.sum(axis=1)
both_gender_selection.iloc[:,1:5].sum(axis=1)

# Histogram 

hist_boy = [1547,762,80,5,1697,40,33,338,13,631]
hist_girl = [58,16,4,0,140,5,0,19,0,55]
hist_both_gender = [1605,777,84,5,1837,45,33,357,13,686]

####################################################################################################

# Plotting histogram


boy_bar_chart = plt.hist(hist_boy,rwidth= 0.9, color='cyan', label='boy' )
plt.legend() 

girl_bar_chart = plt.hist(hist_girl,rwidth= 0.9, color='pink', label='girl' )
plt.legend() 

both_gender_bar_chart = plt.hist(hist_both_gender,rwidth= 0.8, color='green', label='both gender' )
plt.legend()

plt.hist([hist_boy,hist_girl,hist_both_gender],rwidth= 0.75, color=['cyan','pink','green'],
         label=['boy','girl','both gender'],orientation='horizontal' )

plt.legend()

#####################################################################################################
# Pie chart

# exp_chart

boy_exp_vals = [1547,762,80,5,1697,40,33,338,13,631]
girl_exp_vals = [58,16,4,0,140,5,0,19,0,55]
both_gender_exp_vals = [1605,777,84,5,1837,45,33,357,13,686]

# exp_labels

exp_labels = ['Bersabit dengan harta benda','Bersabit dengan orang','Akta Kesalahan Kecil','Melanggar Syarat Pengawasan',
              'Dadah','Judi','Senjata/bahan api','Lalulintas','Lari dari Sekolah Diluluskan','Lain-lain']

# Pie_chart_boy
plt.axis('equal')

Pie_chart_boy = plt.pie(boy_exp_vals, labels=exp_labels, radius=1.75, autopct= '%0.1f%%', shadow=False,
        explode = [0.1,0.1,0.1,0.2,0.25,0.25,0.6,0.1,0.65,0.1], startangle=85)

plt.show(Pie_chart_boy)

# Pie_chart_girl
Pie_chart_girl = plt.pie(girl_exp_vals, labels=exp_labels, radius=1.75, autopct= '%0.1f%%', shadow=False,
        explode = [0.1,0.1,0.1,0.2,0.15,0.25,0.6,0.1,0.65,0.1], startangle=90)

plt.show(Pie_chart_girl)

 # Pie_chart_both_gender
Pie_chart_both_gender = plt.pie(both_gender_exp_vals, labels=exp_labels, radius=1.75, autopct= '%0.1f%%', shadow=False,
         explode = [0.1,0.1,0.1,0.2,0.15,0.25,0.6,0.1,0.65,0.1], startangle=85)

plt.show(Pie_chart_both_gender)

####################################################################################################
st.write(df.head())

st.markdown('_Daripada data set di atas, maka saya cuba mengolah data berkenaan untuk memahami maklumat yang ingin di sampaikan._.')
st.write('Sebentar, adakah anda sudi untuk meneruskan sesi ini ?')

show = st.checkbox('Baiklah...')
if show:
    option = st.radio(
    'Pilih jantina yang anda kehendaki:',
     ['Lelaki','Perempuan','Kedua-duanya'])

    if option=='Lelaki':
        col1, col2 = st.columns(2)
        with col1:
          chart_data = pd.DataFrame(
              np.random.randn(25, 1),
              columns=["a"])
          
          st.bar_chart(chart_data)

        with col2:
          chart_data = pd.DataFrame(
              np.random.randn(20, 3),
              columns=['a', 'b', 'c'])
          st.line_chart(chart_data)


    elif option=='Perempuan':
          col1, col2 = st.columns(2)
          with col1:
            chart_data = pd.DataFrame(
              np.random.randn(10, 1),
              columns=['a'])
          st.bar_chart(chart_data)

          with col2:
            chart_data = pd.DataFrame(
              np.random.randn(20, 3),
              columns=['a', 'b', 'c'])
          st.line_chart(chart_data)



    else:

        st.write('Untuk meneruskan sesi, sila rujuk pada [terma dan syarat] https://www.youtube.com/static?template=terms')
        show = st.checkbox('Ok aje, takde masa dah ni...')
        if show:
            st.video('https://www.youtube.com/watch?v=V67pVb3o4nc', format="video/mp4", start_time=0)
            st.write("Terpaksalah, asyik 'script error' sahaja. Harap maaf, terima kasih...")
            st.markdown('**_Bye, bye..._**')

