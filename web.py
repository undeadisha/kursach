import streamlit as st
from generation_att import *

st.markdown('## Генерирую текстовое описание изображений')

image = st.file_uploader("Загрузите изображение", type=['png', 'jpg','jpeg'])
if image is not None:
    st.image(image)
    img = Image.open(image)

    st.markdown('Результат генерации описания (модель LSTM + MobileNet + Attention):')
    text_att = predict_att(img)
    print(text_att )
    st.markdown('#### __' + text_att + '__')