import tensorflow as tf
model = tf.keras.models.load_model('my_model.h5')

import streamlit as st
st.write("""
# Identifikasi Penyakit Paru
""")
st.write("Identifikasi Penyakit Paru Paru Berdasarkan Citra Foto Thorax Menggunakan CNN")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


from PIL import Image
import numpy as np



dimensi_gambar = (224,224)
channel = (3,)
input_shape = dimensi_gambar + channel
labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Undefined']
def preprocess(gambar, dimensi_gambar):
#sebelum preprocess
    nimg = gambar.convert('RGB').resize(dimensi_gambar, resample= 0)
    # print('sebelum preprocess:', gambar)
    #setelah preprocess
    img_arr = (np.array(nimg))/255
    # print('setelah preprocess: \n',img_arr)
    # st.image(img_arr, use_column_width=True)
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

if file is None:
    st.text("Please upload an image file")
else:
    
    gambar = Image.open(file)
    st.image(gambar, use_column_width=True)
    X = preprocess(gambar, dimensi_gambar)
    X = reshape([X])
    y = model.predict(X)

    st.subheader('Hasil')
    for i in range(0,8):
#        st.markdown(labels[i],f"{(np.max(y)*100).astype(int)}")
        st.markdown(f"<span>{(labels[i])} </span>: <span> {(y[0][i]*100).astype(int)}%</span>",unsafe_allow_html=True)
        # st.subheader('Nilai Probabilitas')
