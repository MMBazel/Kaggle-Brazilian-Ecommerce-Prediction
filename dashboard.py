import streamlit as st
import pickle
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

import streamlit.components.v1 as components
import os
from PIL import Image

full_path = os.getcwd()

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df


brazilian_holidays_df = load_data(f"{full_path}/data/raw/Brazilian_Holidays.csv")
brazilian_population_df = load_data(f"{full_path}/data/raw/Population_Brazilian_Cities_V2.csv")
customers_df = load_data(f"{full_path}/data/raw/olist_customers_dataset.csv")
order_items_df = load_data(f"{full_path}/data/raw/olist_order_items_dataset.csv")
payments_df = load_data(f"{full_path}/data/raw/olist_order_payments_dataset.csv")
orders_df = load_data(f"{full_path}/data/raw/olist_orders_dataset.csv")
products_df = load_data(f"{full_path}/data/raw/olist_products_dataset.csv")
sellers_df = load_data(f"{full_path}/data/raw/olist_sellers_dataset.csv")
report_profile_df = load_data(f"{full_path}/data/backup/profile_report.csv")
states_abbrev_df = load_data(f"{full_path}/data/raw/Brazilian_States_Abbrev.csv")
predictions_df = load_data(f"{full_path}/data/backup/predictions.csv")


st.title("🛍️🌐 Predicting Delivery Times Using A Real, Commercial Brazilian E-Commerce Dataset 🇧🇷💰")
st.header("Processing more than 100K rows of data and training a model in <5 min.")
st.subheader("By Mikiko Bazeley (@BazeleyMikiko)")


front_image = Image.open(f"{full_path}/images/product_image.png")
st.image(front_image, caption='An example of a product image')


st.divider()
st.subheader("About This Project")
st.markdown("""<a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/">👉 Original Kaggle Dataset 🔗</a>""", unsafe_allow_html=True,)
st.markdown("""<a href="https://www.pola.rs/">👉 Polars 🔗</a>""", unsafe_allow_html=True,)
st.text("")
st.markdown("For this project, we decided to push the limits of data processing.")
st.markdown("Using the Polars library, in combination with Pandas, we were able to clean, transform, join, and reshape 8 files containing anywhere from **:blue[20 to 100K records]**  in less than 5-6 min. ")
st.markdown('Leveraging the Power of Z BY HP to make your data science work **_🤯mindblowingly🤯_ fast**.')
st.markdown("_(And decreasing the amount of time staring at the screen waiting for your training job to finish.⏳)_")


st.divider()


tab1, tab2, tab3, tab4, tab5 = st.tabs(["⏱️ Clocking","📈 Data Analysis", "🤿 Data Sources","🤖 Model Prediction", "🗃 Data Quality Issues"])



with tab1: 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Size of Orders Dataset", value=orders_df['order_id'].count())
    col2.metric(label="No. of Tables Used", value=8)
    col3.metric(label="Avg Processing Time for The Data & Modeling Pipeline Combined", value='<5 Min')
    col4.metric(label="Number of Cross-Joins & Complex Queries", value='10+')
    
    st.divider()

    with st.container():
        st.subheader("Pipeline & Model Speed Benchmarks")
        
        col5, col6 = st.columns(2)
        with col5:
            st.text("➡️ Scroll right at bottom of table")
            st.dataframe(report_profile_df)
        with col6:
            st.metric(label="Total Time in Mins",value=round(report_profile_df['Time in Mins'].sum(),2))


with tab2:
    with st.container():
        st.subheader("Volume of Orders Over Time (Histogram)")
        orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'],format="%Y-%m-%d %H:%M:%S")
        vol_df = orders_df.resample(rule='M', on='order_purchase_timestamp')['order_id'].count()
        vol_df = vol_df.to_frame().reset_index().rename(columns = {'order_id':'Vol. Orders (Count)',"order_purchase_timestamp":"Order Purchase (Month - Year)"})
        fig = px.bar(vol_df, x="Order Purchase (Month - Year)", y="Vol. Orders (Count)")
        st.plotly_chart(fig)

    
    st.divider()
    with st.container():
        st.subheader("Total Value of Orders by Brazil State (Bar)")
        vol_region_df = pl.DataFrame(orders_df).join(
            pl.DataFrame(customers_df), on="customer_id").select([
            pl.col('customer_state').alias('Customer State'), 
            pl.col('order_id').count().over('customer_state').alias("Vol Orders Per State")
            ]).unique().sort("Vol Orders Per State",descending=True).to_pandas()
        fig2 = px.bar(vol_region_df, x="Customer State", y="Vol Orders Per State")
        st.plotly_chart(fig2)

        vol_region_df = pl.DataFrame(vol_region_df).join(pl.DataFrame(states_abbrev_df), left_on="Customer State", right_on="Code").to_pandas()
        st.dataframe(vol_region_df)

    st.divider()
    with st.container():
        st.subheader("Avg Delivery Duration: Actual vs Predicted Over Time (Line)")
        st.dataframe(predictions_df.head(20))
        predictions_df['order_purchase_timestamp'] = pd.to_datetime(predictions_df['order_purchase_timestamp'],format="%Y-%m-%d %H:%M:%S")


        predicted_delivery_duration = predictions_df.resample(rule='M', on='order_purchase_timestamp')['predictions'].mean()
        
        actual_delivery_duration = predictions_df.resample(rule='M', on='order_purchase_timestamp')['label_actual_delivery_duration'].mean()
        
        original_estimate_delivery_duration = predictions_df.resample(rule='M', on='order_purchase_timestamp')['label_estimated_delivery_duration'].mean()

        all_series = pd.concat([predicted_delivery_duration, actual_delivery_duration, original_estimate_delivery_duration], axis=1).reset_index()
        

        colors = px.colors.qualitative.Plotly
        fig3 = go.Figure()
        fig3.add_traces(go.Scatter(x=all_series['order_purchase_timestamp'], y = all_series['predictions'], mode = 'lines',name='RF Predicted Duration', line=dict(color=colors[0])))
        fig3.add_traces(go.Scatter(x=all_series['order_purchase_timestamp'], y = all_series['label_actual_delivery_duration'], mode = 'lines',name="Actual Duration", line=dict(color=colors[1])))
        fig3.add_traces(go.Scatter(x=all_series['order_purchase_timestamp'], y = all_series['label_estimated_delivery_duration'], mode = 'lines',name="OG Estimated Duration", line=dict(color=colors[2])))
        st.plotly_chart(fig3)





with tab3: 
    st.title("Exploring The Original Data Sources")
    st.markdown("Let's explore the datasets being used to train our regression model.")

    image = Image.open(f"{full_path}/images/kaggle_chart.png")
    st.image(image, caption='ERD of Kaggle\'s Olist Ecommerce Dataset')



    option = st.selectbox('Select dataset to preview',('Brazilian Holidays', 'Brazilian Cities Population', 'Customers', 'Order Items','Payments','Orders','Products','Sellers'))

    st.write('You selected:', option)

    match option:
        case 'Brazilian Holidays':
            st.dataframe(brazilian_holidays_df)
        case 'Brazilian Cities Population':
            st.dataframe(brazilian_population_df)
        case 'Orders':
            st.dataframe(orders_df)
        case 'Customers':
            st.dataframe(customers_df)
        case 'Order Items':
            st.dataframe(order_items_df)
        case 'Payments':
            st.dataframe(payments_df)
        case 'Products':
            st.dataframe(products_df)
        case 'Sellers':
            st.dataframe(sellers_df)


with tab4:
    model = pickle.load(open(f"{full_path}/model/model.pkl",'rb'))

    st.title("Predicting Delivery (In Days)")
    st.markdown("Here we'll predict the delivery times based on a couple factors for a single order.")
    st.markdown("On average the performance of the was around R^2 ~= 0.86")
    st.subheader("Please enter values for the following: ")


    with st.form("try model"):
            feat_total_payment_value = st.number_input('Total Order Value',min_value=1, max_value=20000,key="feat_total_payment_value_DROPDOWN")
            feat_num_items_per_order = st.number_input('No. of Items',min_value=1, max_value=30,key="feat_num_items_per_order_DROPDOWN")
            feat_num_cat_per_order = st.number_input('No. of Product Categories',min_value=1, max_value=5,key="feat_num_cat_per_order_DROPDOWN")
            feat_num_sellers_per_order = st.number_input('No. of Sellers',min_value=1, max_value=5,key="feat_num_sellers_per_order_DROPDOWN")
            feat_num_seller_cities_per_order = st.number_input('No. Seller Locations',min_value=1, max_value=5,key="feat_num_seller_cities_per_order_DROPDOWN")
            feat_customer_holiday_impact = st.number_input('No. Current Holidays In Customer Region',min_value=0, max_value=3,key="feat_customer_holiday_impact_DROPDOWN")
            feat_seller_holiday_impact = st.number_input('No. Current Holidays In All Seller Regions',min_value=0, max_value=15,key="feat_seller_holiday_impact_DROPDOWN")

            submitted = st.form_submit_button("Submit")
            if submitted:
                st.subheader("Predicted Delivery Duration in Days")
                prediction = model.predict([[feat_total_payment_value, feat_num_items_per_order, feat_num_cat_per_order, feat_num_sellers_per_order, feat_num_seller_cities_per_order, feat_customer_holiday_impact,feat_seller_holiday_impact ]])/24.0
                st.code(float(prediction))




with tab5:
    st.title("Understanding The Dataset")
    st.markdown("Here we use pandas-profiler to understand the relationships between the different features of our model and where we can improve.")

    EDA_df = load_data(f"{full_path}/data/backup/exploratory_data_set.csv")

    @st.cache_resource
    def get_profile_report():
        return EDA_df.profile_report() 

    st_profile_report(get_profile_report())


        
        