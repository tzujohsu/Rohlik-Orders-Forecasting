import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_train_dist_by_warehouse(train_df: pd.DataFrame):
    fig, axes = plt.subplots(7,3, figsize=(14,7*3), gridspec_kw={'width_ratios':[0.70, 0.30, 0.20]})

    for i,(ind, c) in enumerate(train_df.groupby('warehouse')):
        ax = axes[i,0]
        sns.lineplot(data=c,x=c.date,y='orders',ax=ax, linewidth = 0.7)
        ax.set_title(f'{ind}')
        ax.tick_params(axis='x', rotation=90)


        ax = axes[i,1]
        sns.histplot(data=c,x='orders',ax=ax,color='green', linewidth = 0.7)
        ax.set_title(f'{ind} - skew ({c.orders.skew().round(2)})')

        ax = axes[i,2]
        sns.boxplot(data=c,y='orders',ax=ax,color='orange', linewidth=.65, fliersize=2.01, width=0.70)

    plt.tight_layout()

def display_mean_order(
        df:pd.DataFrame
        , warehouse:str
        , month_list:list) -> None:
    """
    Function to plot mean orders by days in the week for each warehouse
    
    """
    # Filter the DataFrame for select warehouse
    prague_df = df[df['warehouse'] == warehouse]

    prague_df = prague_df[prague_df['date'].dt.month.isin(month_list)]

    # Extract year and day of the week
    prague_df['year'] = prague_df['date'].dt.year
    prague_df['day_of_week'] = prague_df['date'].dt.day_name()

    # Calculate mean orders for each day of the week for each year
    mean_orders_by_day_year = prague_df.groupby(['year', 'day_of_week'])['orders'].mean().unstack()

    teamp_days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_order = []
    for day in teamp_days_order:
        if day in mean_orders_by_day_year.columns:
            days_order.append(day)
    mean_orders_by_day_year = mean_orders_by_day_year[days_order]

    plt.figure(figsize=(10, 4))
    for year in mean_orders_by_day_year.index:
        plt.plot(mean_orders_by_day_year.columns, mean_orders_by_day_year.loc[year], marker='o', linestyle='-', label=f'Year {year}')

    plt.title(f'Mean Orders by Day of the Week for {warehouse}')
    plt.xlabel('Day of the Week')
    plt.ylabel('Mean Orders')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

def display_predictions(
        target_column
        , years:list
        , plot_continue:bool
        , train_df: pd.DataFrame
        , force_holiday:str
        , use_submission:bool
        , submission_df: pd.DataFrame=None) -> None:
    """
    Plot full timeline including predictions.

    target_column: Column name for the target variable.
    years: years to plot.
    plot_continue: ignore, have some bug now.
    train_df: Train data.
    force_holiday: Zoom to date around this day for easy to observe.
    use_submission: Include was predict orders in plot.
    submission_df: You last submission dataframe. DataFrame containing the predictions. Must also include warehouse names.
    """
    if use_submission:
        submission_df = submission_df.rename(columns={"id": "warehouse"})
        submission_df['date'] = submission_df['warehouse'].str.split('_').str[2]
        submission_df['warehouse'] = submission_df['warehouse'].str.split('_').str[:2].str.join('_')
        submission_df['date'] = pd.to_datetime(submission_df['date'], errors='coerce')

    train_df['date'] = pd.to_datetime(train_df['date'], errors='coerce')

    if force_holiday:
        plot_continue=True
        for year in years:
            plt.figure(figsize=(15, 6))
            for wh in train_df["warehouse"].unique():
                wh_train = train_df[(train_df["warehouse"] == wh) & (train_df['date'].dt.year==year)][["warehouse", target_column, "date", "holiday", "holiday_name"]].set_index("date")
                if use_submission:
                    wh_submission = submission_df[submission_df["warehouse"] == wh][["warehouse", target_column, "date"]].set_index("date")

                if force_holiday:
                    # Get the dates of the force_holiday
                    holiday_dates = wh_train[wh_train['holiday_name'] == force_holiday].index

                    if holiday_dates.empty:
                        continue

                    # Consider only the first occurrence of the force_holiday
                    holiday_date = holiday_dates[0]
                    start_date = holiday_date - pd.Timedelta(days=7)
                    end_date = holiday_date + pd.Timedelta(days=3)

                    wh_train = wh_train[(wh_train.index >= start_date) & (wh_train.index <= end_date)]
                    if use_submission:
                        wh_submission = wh_submission[(wh_submission.index >= start_date) & (wh_submission.index <= end_date)]

                if plot_continue:
                    # Plot all years in one line
                    plt.plot(wh_train.index, wh_train[target_column], label=wh, linewidth=1)
                    if use_submission:
                        plt.plot(wh_submission.index, wh_submission[target_column], label='Predictions', color='red', linewidth=0.8)
                else:
                    # Plot each year as one line
                    for year in years:
                        wh_train_year = wh_train[wh_train.index.year == year]
                        plt.plot(wh_train_year.index, wh_train_year[target_column], label=f'Train Data {year}', linewidth=0.8)
                        if use_submission:
                            wh_submission_year = wh_submission[wh_submission.index.year == year]
                            plt.plot(wh_submission_year.index, wh_submission_year[target_column], label=f'Predictions {year}', linewidth=0.8)

                # Add vertical lines for holidays
                for holiday_date in holiday_dates:
                    plt.axvline(x=holiday_date, linestyle='--', linewidth=0.6, label=f'{force_holiday}_{wh}' if holiday_date == holiday_dates[0] else "")

            plt.title(f"Orders for holiday {year}")
            plt.xlabel("Date")
            plt.ylabel(target_column)
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.show()
    else:
        for wh in train_df["warehouse"].unique():
            wh_train = train_df[train_df["warehouse"] == wh][["warehouse", target_column, "date", "holiday", "holiday_name"]].set_index("date")
            if use_submission:
                wh_submission = submission_df[submission_df["warehouse"] == wh][["warehouse", target_column, "date"]].set_index("date")

            plt.figure(figsize=(15, 6))

            if plot_continue:
                # Plot all years in one line
                plt.plot(wh_train.index, wh_train[target_column], label='Train Data', color='blue', linewidth=0.8)
                if use_submission:
                    plt.plot(wh_submission.index, wh_submission[target_column], label='Predictions', color='red', linewidth=0.8)
            else:
                # Plot each year as one line
                for year in years:
                    wh_train_year = wh_train[wh_train.index.year == year]
                    plt.plot(wh_train_year.index, wh_train_year[target_column], label=f'Train Data {year}', linewidth=0.8)
                    if use_submission:
                        wh_submission_year = wh_submission[wh_submission.index.year == year]
                        plt.plot(wh_submission_year.index, wh_submission_year[target_column], label=f'Predictions {year}', linewidth=0.8)

            # Add vertical lines for holidays
            holiday_dates = wh_train[wh_train['holiday'] == 1].index
            for holiday_date in holiday_dates:
                plt.axvline(x=holiday_date, color='green', linestyle='--', linewidth=0.6, label='Holiday' if holiday_date == holiday_dates[0] else "")

            # # Add vertical lines for 'Good Friday'
            # good_friday_dates = wh_train[wh_train['holiday_name'] == 'good friday'].index
            # for good_friday_date in good_friday_dates:
            #     plt.axvline(x=good_friday_date, color='red', linestyle='--', linewidth=0.8, label='Good Friday' if good_friday_date == good_friday_dates[0] else "")

            # # Add vertical lines for all Fridays
            # fridays = wh_train[wh_train.index.dayofweek == 4].index
            # for friday in fridays:
            #     plt.axvline(x=friday, color='purple', linestyle='--', linewidth=0.7, label='Friday' if friday == fridays[0] else "")

            # # Add vertical lines for days before holidays
            # day_before_holiday_dates = (wh_train.index[wh_train['holiday'] == 1] - pd.Timedelta(days=1)).intersection(wh_train.index)
            # for day_before_holiday in day_before_holiday_dates:
            #     plt.axvline(x=day_before_holiday, color='orange', linestyle='--', linewidth=0.7, label='Day before holiday' if day_before_holiday == day_before_holiday_dates[0] else "")

            plt.title(f"Orders for warehouse {wh}")
            plt.xlabel("Date")
            plt.ylabel(target_column)
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.show()


