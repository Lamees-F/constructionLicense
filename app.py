import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta

model_daily = joblib.load("xgb_permits_daily_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="أمانة منطقة عسير", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            direction: rtl;
            text-align: right;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* تنسيق sidebar */
        .css-1d391kg {
            direction: rtl;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("نبذة عن المشروع")
st.sidebar.info("""
هذا التطبيق يتيح لك توقع عدد طلبات الرخص الإنشائية التي قد تصدرها أمانة منطقة عسير بناءً على بيانات السنوات السابقة.
يمكنك اختيار تاريخ بداية التوقع والفترة (يومي، أسبوعي، شهري) لمعرفة التوقعات التفصيلية التي تساعد في التخطيط العمراني والاستثماري.

**مصدر البيانات:** سجلات الرخص الإنشائية من أمانة منطقة عسير.

**كيفية الاستخدام:**
- اختر تاريخ بداية التوقع.
- اختر الفترة الزمنية.
- حدد الخصائص المطلوبة.
- اضغط زر "توقع" للحصول على النتائج.
""")

st.title("توقع عدد طلبات الرخص الإنشائية في منطقة عسير")


start_date = st.date_input("اختر تاريخ بداية التوقع")

period = st.selectbox("اختر فترة التوقع:", ['يومي', 'أسبوعي', 'شهري'])

municipality = st.selectbox("اختر البلدية:", encoders['البلدية'].classes_)
request_type = st.selectbox("اختر الغرض:", encoders['الغرض'].classes_)
# immediate = st.selectbox("فوري أو غير فوري:", encoders['فوري / غير فوري'].classes_)
ownership_type = st.selectbox("نوع سند الملكية:", encoders['نوع سند الملكية'].classes_)
# main_use = st.selectbox("استخدام المبنى الرئيسي:", encoders['إستخدام المبنى الرئيسي'].classes_)
# license_duration = st.selectbox("فئة مدة صلاحية الرخصة:", encoders['فئة مدة صلاحية الرخصة'].classes_)

try:
    municipality_code = encoders['البلدية'].transform([municipality])[0]
    request_type = encoders['الغرض'].transform([request_type])[0]
    # immediate_code = encoders['فوري / غير فوري'].transform([immediate])[0]
    ownership_code = encoders['نوع سند الملكية'].transform([ownership_type])[0]
    # main_use_code = encoders['إستخدام المبنى الرئيسي'].transform([main_use])[0]
    # license_duration_code = encoders['فئة مدة صلاحية الرخصة'].transform([license_duration])[0]
except Exception as e:
    st.error(f"خطأ في التشفير: {e}")
    st.stop()

def create_input_df(date):
    year = date.year
    month = date.month
    day = date.day
    week = date.isocalendar()[1]
    day_of_week = date.weekday() 

    data = {
        'البلدية': [municipality_code],
        'الغرض': [request_type],
        # 'فوري / غير فوري': [immediate_code],
        'نوع سند الملكية': [ownership_code],
        # 'إستخدام المبنى الرئيسي': [main_use_code],
        # 'فئة مدة صلاحية الرخصة': [license_duration_code],
        'سنة الطلب': [year],
        'شهر الطلب': [month],
        'أسبوع الطلب': [week],
        'يوم الطلب': [day],
        'يوم الأسبوع': [day_of_week]
    }
    return pd.DataFrame(data)

if st.button("توقع"):
    if period == 'يومي':
        input_df = create_input_df(start_date)
        pred = model_daily.predict(input_df)[0]
        if pred < 0:
            st.warning("⚠️ لا توجد طلبات متوقعة لهذا اليوم.")
        else:
            st.success(f"🔹 توقع عدد طلبات الرخص ليوم {start_date.strftime('%Y-%m-%d')} هو **{round(pred)}** طلب.")
    
    else:
        if period == 'أسبوعي':
            days_to_predict = 7
            label = f"الأسبوع من {start_date.strftime('%Y-%m-%d')} إلى {(start_date + timedelta(days=6)).strftime('%Y-%m-%d')}"
        else:  
            next_month = start_date.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            days_to_predict = (last_day - start_date).days + 1
            label = f"الشهر من {start_date.strftime('%Y-%m-%d')} إلى {last_day.strftime('%Y-%m-%d')}"
        
        total_pred = 0
        daily_preds = []
        for i in range(days_to_predict):
            current_date = start_date + timedelta(days=i)
            input_df = create_input_df(current_date)
            pred = model_daily.predict(input_df)[0]
            daily_preds.append((current_date.strftime('%Y-%m-%d'), pred))
            total_pred += pred
        
        if total_pred < 0:
            st.warning("⚠️ لا توجد طلبات متوقعة لهذه الفترة.")
        else:
            st.success(f"🔹 توقع إجمالي عدد طلبات الرخص للفترة {label} هو **{round(total_pred)}** طلب.")
            
            with st.expander("عرض التفاصيل اليومية"):
                for date_str, val in daily_preds:
                    if val < 0:
                        st.write(f"{date_str}: لا توجد طلبات")
                    else:
                        st.write(f"{date_str}: {round(val)} طلب")

