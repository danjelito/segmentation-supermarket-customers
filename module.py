import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def drop_outliers(df: pd.DataFrame, columns: list, method: str):
    """Drop outliers from a dataframe

    Args:
        df (pd.DataFrame): dataframe to drop outliers from
        columns (list): list of columns to check
        method (str): methid to use

    Returns:
        pd.DataFrame: dataframe with outliers dropped
    """

    index_outliers= []

    if method == 'z_score':
        for col in columns:
            lower_limit = df[col].mean() - 3*df[col].std()
            upper_limit = df[col].mean() + 3*df[col].std()
            indexes= df.loc[
                (df[col] < lower_limit) | (df[col] > upper_limit)
            ].index.to_list()
            index_outliers.extend(indexes)
    else:
        print('method unrecognized')

    df= df.drop(index= index_outliers)
    return df


def plot_scatter_with_centroids(x: str, y: str, scaler, model, df):
    """Scatter plot with centroids marked

    Args:
        x (str): x-axis
        y (str): y-axis
        scaler (_type_): scaler object
        model (_type_): model object 
        df (_type_): dataframe with features
    """

    fig, ax= plt.subplots(figsize= (12, 8), dpi= 150, facecolor= 'white')

    centroids= (pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns= pd.get_dummies(df.drop(columns= 'segment')).columns
        )
        [[x, y]]
    )
    sns.scatterplot(
        data= df, x= x, y= y, hue= 'segment', ax= ax
    )

    sns.scatterplot(
        data= centroids, x= x, y= y, color= 'black', 
        marker= 'x', s= 1000, ax= ax
    )
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(f'{y.title().replace("_", " ")} vs {x.title().replace("_", " ")}', 
        fontweight= 'bold', pad= 30)
    ax.set_xlabel(x.title().replace('_', ' '), fontweight= 'bold')
    ax.set_ylabel(y.title().replace('_', ' '), fontweight= 'bold')
    sns.move_legend(ax, "upper center", title= None, ncol= 5, 
        bbox_to_anchor= (0.5, 1.05))
    ax.set_facecolor('white')
    plt.show()


def plot_violin(x: str, y: str, df: pd.DataFrame):
    """Violin plot

    Args:
        x (str): segment
        y (str): feature
        df (pd.DataFrame)
    """

    fig, ax= plt.subplots(figsize= (12, 4), dpi= 150)

    sns.violinplot(x= df[x], y= df[y], ax= ax, palette= 'tab10', alpha= 0.8)

    ax.set_title(f'{y.title()} by {x.title()}', fontweight= 'bold')
    ax.set_xlabel(x.title(), fontweight= 'bold')
    ax.set_ylabel(y.title(), fontweight= 'bold')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()

def plot_bar(x, y, df, ylim= None):
    """Bar plot per segment

    Args:
        x (str): x-axis
        y (str): y-axis
        df (pd.DataFrame)
        ylim (tuple, optional): y-lim if any. Defaults to None.
    """

    fig, ax= plt.subplots(figsize= (12, 4), dpi= 150)

    sns.barplot(
        x= df[x], y= df[y], palette= 'Greens', ax= ax
    )
    ax.set_title(f'{y.title().replace("_", " ")} by {x.title()}', fontweight= 'bold')
    ax.set_xlabel(x.title().replace("_", " "), fontweight= 'bold')
    ax.set_ylabel(y.title().replace("_", " "), fontweight= 'bold')
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis= 'x', visible= False)
    if ylim:
        ax.set_ylim(ylim)
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), '.1f'), 
            (p.get_x() + p.get_width() / 2., p.get_height()),  
            ha = 'center', va = 'center', 
            xytext = (0, 10), textcoords = 'offset points'
        )
    plt.show()

def plot_bar_with_hue(hue: str, df: pd.DataFrame):
    """Bar plot per segment with hue

    Args:
        hue (str): feature to divide per segment
        df (pd.DataFrame): dataframe to process
    """

    x= 'segment'
    y= 'percentage'

    data= (df
        .groupby([x, hue])
        .agg(
            count= (hue, 'size')
        )
        .assign(
            percentage= lambda df_: (
                df_['count'] / df_.groupby(x)['count'].transform('sum')
            ).round(3)
        )
        .reset_index()
    )

    fig, ax= plt.subplots(figsize= (12, 4), dpi= 150)
    ax= sns.barplot(
        x= data[x], 
        y= data[y], 
        hue= data[hue],
        palette= 'Greens'
    )
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), '.1%'), 
            (p.get_x() + p.get_width() / 2., p.get_height()),  
            ha = 'center', va = 'center', 
            xytext = (0, 10), textcoords = 'offset points'
        )

    hue= hue.replace('_', ' ')
    ax.set_title(f'{hue.title()} by {x.title()}', fontweight= 'bold')
    ax.set_xlabel(x.title(), fontweight= 'bold')
    ax.set_ylabel(y.title(), fontweight= 'bold')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(title= None)
    plt.show()

def plot_kde_box(x: str, y: str, df):
    """KDE plot and boxplot by segment

    Args:
        x (str): feature to divide by segment
        y (str): segment
        df (pd.DataFrame): dataframe to process
    """

    fig, ax= plt.subplots(nrows= 2, figsize= (12, 8), sharex= True, dpi= 150)

    sns.kdeplot(
        data= df, 
        x= x, 
        hue= y,
        fill= True,
        ax= ax[0]
    )
    sns.boxplot(
        data= df, 
        x= x, 
        y= y,
        saturation= 0.8, 
        ax= ax[1]
    )
    plt.suptitle(f'{x.title().replace("_", " ")} by {y.title()}', fontweight= 'bold', y= 0.95)
    ax[1].set_xlabel(x.title().replace("_", " "), fontweight= 'bold')
    ax[0].set_ylabel('Density', fontweight= 'bold')
    ax[1].set_ylabel(y.title(), fontweight= 'bold')
    sns.move_legend(ax[0], "upper center", title= y.title(), ncol= 5, 
        bbox_to_anchor= (0.5, 1.1))
    for i in (0, 1):
        ax[i].spines[['right', 'top']].set_visible(False)
        # ax[i].legend(title= None)
    plt.show()

def plot_heatmap(df: pd.DataFrame, title: str):
    """Plot heatmap of percentage with annotation

    Args:
        df (pd.DataFrame)
        title (str): plot title
    """

    fig, ax= plt.subplots(figsize= (12, 4), dpi= 150)
    sns.heatmap(
        df, 
        cmap= 'Greens', 
        annot= True,
        fmt= '.1%', 
        cbar= False,
        ax= ax
    )
    ax.set_title(title, fontweight= 'bold')
    ax.set_ylabel('Segment', fontweight= 'bold')
    xlabels= [x
        .get_text()
        .replace('mnt', '')
        .replace('_', ' ')
        .split(' ')[1]
        .strip()
        .title()
        for x in ax.get_xticklabels()
    ]
    ax.set_xticklabels(xlabels)

    plt.show()