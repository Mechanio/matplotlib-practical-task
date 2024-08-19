import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("AB_NYC_2019.csv")
df.fillna(value={'name': 'Unknown', 'host_name': 'Unknown', 'last_review': pd.NaT}, inplace=True)  # Fill missing values
df = df[df['price'] > 0]


def neightbourhood_distr_of_listing(dataframe):
    neigbourhoods_count = dataframe['neighbourhood_group'].value_counts()
    plt.title('Neighborhood Distribution of Listings')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Number of Listings')
    bars = plt.bar(neigbourhoods_count.index, neigbourhoods_count.values,
                   color=['orange', 'pink', 'palegreen', 'yellow', 'cyan'])
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()), va='bottom', ha='center')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('1-neightbourhood_distr_of_listing.png')
    plt.show()


def price_distr_across_neighborhoods(dataframe):
    price_distr = dataframe.groupby('neighbourhood_group')['price'].apply(list).to_dict()
    plt.title('Price Distribution Across Neighborhoods')
    plt.xlabel('Neighborhood Group')
    plt.ylabel('Price (Log Scale)')
    bplot = plt.boxplot(list(price_distr.values()), tick_labels=list(price_distr.keys()), patch_artist=True)
    for patch, color in zip(bplot['boxes'], ['orange', 'pink', 'palegreen', 'yellow', 'cyan']):
        patch.set_facecolor(color)
    plt.ylim(10, 500)
    plt.grid()
    plt.savefig('2-price_distr_across_neighborhoods.png')
    plt.show()


def room_type_availability(dataframe):
    price_distr = pd.pivot_table(dataframe, index='neighbourhood_group', columns='room_type', values='availability_365',
                                 aggfunc="mean")
    x = np.array([0, 0.25, 0.5])  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title('Average Availability by Room Type Across Neighborhoods')
    ax.set_xlabel('Neighborhood Group')
    ax.set_ylabel('Average Availability ')
    for group in range(len(price_distr.values)):
        offset = width + multiplier
        rects = ax.bar(x + offset, price_distr.values[group], width, color=['orange', 'palegreen', 'cyan'],
                       label=price_distr.keys())
        ax.bar_label(rects, padding=3, fmt='%.1f')
        multiplier += 1

    ax.set_xticks(np.array([0.5, 1.5, 2.5, 3.5, 4.5]), price_distr.index)
    ax.legend(handles=rects, loc='upper left', ncols=3)
    plt.savefig('3-room_type_availability.png')
    plt.show()


def price_num_of_reviews_correlation(dataframe):
    price_review_cor = dataframe.groupby('room_type')[['price', 'number_of_reviews']]
    for room_type, data in price_review_cor:
        plt.scatter(data["price"], data["number_of_reviews"], label=room_type, alpha=0.6)
        m, b = np.polyfit(data['price'], data['number_of_reviews'], 1)
        plt.plot(data['price'], data['price']*m+b, linestyle='--')
    plt.title('Correlation Between Price and Number of Reviews')
    plt.xlabel('Price')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Room Type')
    plt.grid()
    plt.savefig('4-price_num_of_reviews_correlation.png')
    plt.show()


def analysis_of_reviews(dataframe):
    dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])
    df_grouped = dataframe.groupby(['neighbourhood_group', 'last_review'])['number_of_reviews'].sum().reset_index()
    # Create a rolling average
    df_grouped['rolling_avg'] = df_grouped.groupby('neighbourhood_group')['number_of_reviews'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean())

    for neighbourhood_group in df_grouped['neighbourhood_group'].unique():
        subset = df_grouped[df_grouped['neighbourhood_group'] == neighbourhood_group]
        plt.plot(subset['last_review'], subset['rolling_avg'], label=neighbourhood_group)

    plt.title('Trend of Number of Reviews Over Time by Neighbourhood Group')
    plt.xlabel('Date of Last Review')
    plt.ylabel('Number of Reviews (Rolling Average)')
    plt.legend(title='Neighbourhood Group')
    plt.grid()
    plt.savefig('5-analysis_of_reviews.png')
    plt.show()


def price_availability_heatmap(dataframe):
    labels = ['0-60', '61-120', '121-180', '181-240', '241-300', '301-365']
    dataframe['availability_bins'] = pd.cut(dataframe['availability_365'], bins=[0, 60, 120, 180, 240, 300, 365],
                                            labels=labels, include_lowest=True)

    df_agg = dataframe.groupby(['neighbourhood_group', 'availability_bins']).agg({
        'price': 'mean',
    }).reset_index()

    pivot_table = df_agg.pivot_table(index='neighbourhood_group', columns='availability_bins', values='price')
    plt.imshow(pivot_table, aspect='auto', cmap='YlGnBu', origin='lower')

    cbar = plt.colorbar()
    cbar.set_label('Average Price')
    plt.title('Heatmap of Price vs. Availability Across Neighborhoods')
    plt.xlabel('Availability')
    plt.ylabel('Neighborhood')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(pivot_table.index.size), labels=pivot_table.index)
    plt.tight_layout()
    plt.savefig('6-price_availability_heatmap.png')
    plt.show()


def room_type_review(dataframe):
    review_distr = pd.pivot_table(dataframe, index='neighbourhood_group', columns='room_type', values='number_of_reviews',
                                 aggfunc='sum')
    review_distr.plot(kind='bar', stacked=True)
    plt.title('Number of Reviews by Room Type and Neighbourhood Group')
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.savefig('7-room_type_review.png')
    plt.show()


neightbourhood_distr_of_listing(df)
price_distr_across_neighborhoods(df)
room_type_availability(df)
price_num_of_reviews_correlation(df)
analysis_of_reviews(df)
price_availability_heatmap(df)
room_type_review(df)
