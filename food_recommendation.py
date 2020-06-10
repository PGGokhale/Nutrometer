import pandas as pd
import numpy as np
import random

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

global df_for_cluster 
df_for_cluster = pd.read_csv("db/nutrition_prediction.csv")

#drop the leaving agents as they cannot be consumed as food portions
indices = [i for i in df_for_cluster.index if (i >= df_for_cluster.index[df_for_cluster["NDB_No"] == 18369].values[0]) and ( i <= df_for_cluster.index[df_for_cluster["NDB_No"] == 18375].values[0])]
df_for_cluster = df_for_cluster.drop(indices, axis=0)

# Dropped the baby food items
df_for_cluster = df_for_cluster.drop(df_for_cluster.index[df_for_cluster["FdGrp_Cd"] == 300], axis=0)

# Dropped the spices from the table as they cannot be consumed as food portions
df_for_cluster = df_for_cluster.drop(df_for_cluster.index[df_for_cluster["FdGrp_Cd"] == 200], axis=0)

# Recalculating the nutrients per calorie
Required_columns = ['Calcium', 'Carbohydrate', 'Choline', 'Cobalamin_VB12', 'Copper',
         'Fiber', 'Folate_VB9',  'Iron',
       'Magnesium', 'Manganese',  'Niacin_VB3',
       'Pantothenic_Acid_VB5', 'Phosphorus', 'Potassium', 'Protein',
       'Pyridoxine_VB6', 'Riboflavin_VB2', 'Selenium',  'Sodium',
       'Thiamine_VB1', 'Vitamin_A', 'Vitamin_C', 'Vitamin_D', 'Vitamin_E',
       'Vitamin_K', 'Water',  'Zinc']

for column in Required_columns:
    df_for_cluster[f"{column}/cal"] = df_for_cluster[column]/df_for_cluster["Energy"]

df_for_cluster = df_for_cluster.dropna(how='any',axis=0)


def Findtheclusters(element, n_clus, isHigh, threshold):

    print(element)

 # Use the element/cal columns for K-mean clustering
    element_per_cal = [f"{item}/cal" for item in element]

    X= pd.get_dummies(
        df_for_cluster[[column for column in df_for_cluster.columns if column in element_per_cal]], drop_first = True
    ).values

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    kmeans = KMeans(n_clusters=n_clus,n_init = 10, random_state=0).fit(X_scaled)

    element.append('NDB_No')

    average_nutrition = list()
    Df_cluster_list = list()

    #Once the clusters are identified get one dataframes per cluster     
    for i in range(n_clus):
        indices = np.array(np.argwhere(kmeans.labels_== i).flatten())
        df_new = pd.DataFrame()
       
        for item in indices:
            df_new = df_new.append(df_for_cluster.loc[(df_for_cluster.index==item),element], ignore_index=True)

        average_nutrition.append(df_new[element].sum()[0]/len(df_new.index))

        Df_cluster_list.append(df_new)
    midvalue = [x for x in average_nutrition if ((x != max(average_nutrition)) and (x!=min(average_nutrition)))]
    
    mid_cluster_index = average_nutrition.index(midvalue[0])
    low_cluster_index = average_nutrition.index(min(average_nutrition))
    high_cluster_index = average_nutrition.index( max(average_nutrition))

    #based on the threshold input return list of NDB_No for an appropriate cluster dataframe
    if (isHigh):
        if threshold > 0.25:
            if (len(Df_cluster_list[high_cluster_index]) <=3):
                if (len(Df_cluster_list[mid_cluster_index]) <=3):
                    return Df_cluster_list[low_cluster_index]["NDB_No"]
                else:
                    return Df_cluster_list[mid_cluster_index]["NDB_No"]
            else:
                return Df_cluster_list[high_cluster_index]["NDB_No"]
        else:
            if (len(Df_cluster_list[mid_cluster_index]) <=3):
                return Df_cluster_list[low_cluster_index]["NDB_No"]
            else:
                return Df_cluster_list[mid_cluster_index]["NDB_No"]
            
    else:
        return Df_cluster_list[low_cluster_index]["NDB_No"]


def calcSum(basket):
  
    sum_0 = basket.sum()   
    return sum_0

def calcScore(target, nutsSum):
    score = list()

    for i in range(len(target)):
        diff = target[i]-nutsSum[i]           
        score.append(diff*diff)
    return score



# def hillClimbing(nutrients, displaylist, target, items_in_basket):
def hillClimbing(inputs_to_function):

    nutrients = inputs_to_function["first"]
    displaylist = inputs_to_function["second"]
    target = inputs_to_function["third"]
    items_in_basket = inputs_to_function["fourth"]
  
    basket = pd.DataFrame()
    basket_NDB  = pd.DataFrame()
    sum_C =0
    currentScore = 10001
    minScore = currentScore
    used_index = list()
    NoBasketEnteries = items_in_basket
    Norm_score_list_sum = [1 for i in range(len(nutrients))]
    iteration =0
    max_no_iterations = 15
    max_squared_error = calcScore(target, [0 for i in range(len(nutrients))])
    iteration_inner_loop = 0
    stuck_in_loop = False
   
    
    while True:
        
        # to track number of iterations of external while loop
        iteration += 1
        print(f"Iteration : {iteration}")
        if(iteration == max_no_iterations):
            print(f"Minimum score : {minScore}")
            print(basket_NDB)     
            break
        
        #sum the contents of the basket to get the total amount of nutrients and score
        sum_C = calcSum(basket)
        if(iteration == 1):
            Score_total = calcScore(target, [0 for i in range(len(nutrients))])
            sum_C=[0 for i in range(len(nutrients))]
        else:
            Score_total = calcScore(target, sum_C)
        
        #find the normalized error of the nutrients
        diff = list()      
        abs_diff = list()
        sign = lambda a: 1 if a>0 else -1 if a<0 else 0


        for i in range(len(target)):
            diff.append((target[i]-sum_C[i])/target[i])
            abs_diff.append(abs((target[i]-sum_C[i])/target[i]))
            
        #find the nutrient whose error is maximum. If the nutrient exceeds the required value then low nutrient cluster to be used for selection else moderate cluster    
        # if the loop is stuck then choose a nutrient with minimum error. Sometimes if a particular nutrient has large error then a particular food gets selected
        # repeatedly and the loop gets stuck. Hence to avaoid loop getting stuck the strategy is changed. 
        if stuck_in_loop:
            stuck_in_loop = False
            use_element = abs_diff.index(min(abs_diff))
        
        else:
            max_nutrient_error = abs_diff.index(max(abs_diff))
            use_element = max_nutrient_error
        
        if (sign(diff[use_element]) == -1):
            High_cluster = False
        else:
            High_cluster = True

        print(abs_diff[use_element])       
        
        selected_list_food = Findtheclusters([nutrients[use_element]], 3, High_cluster, abs_diff[use_element])
        print(selected_list_food)
        while True:
            index = random.choice(selected_list_food)
            
            print(index)
            

            if(index in used_index) and (iteration_inner_loop<2):
                iteration_inner_loop = iteration_inner_loop + 1
                if iteration_inner_loop == 2:
                    used_index.append(index)
                    break
                elif (iteration_inner_loop < 6):
                    continue
                else:
                    stuck_in_loop = True
                    iteration_inner_loop = 0
                    break
            else:
                iteration_inner_loop = 0
                used_index.append(index)
                break
        if stuck_in_loop:
            iteration = iteration-1
            break
        food1 = df_for_cluster.loc[df_for_cluster["NDB_No"] == index, nutrients]
        food1_NDB = df_for_cluster.loc[df_for_cluster["NDB_No"] == index, displaylist]
        prev_score = currentScore
        if (minScore > prev_score):
            minScore = prev_score
        
        Score_list = list()
        score_list_sum = list()
        if(minScore > 500):
            basket = basket.append(food1, ignore_index=True)
            basket_NDB = basket_NDB.append(food1_NDB,ignore_index=True )
            
            # Calculate the total nutrients by adding the corresponding nutrients of all food items in the basket
            sum_C = calcSum(basket)

            # Find normalized sum squared error for each of the nurtient in the basket
            for i in range(len(basket.index)):
                
                basket_list =basket.values.tolist()
                temp = [k / j for k, j in zip(calcScore(target, basket_list[i]), max_squared_error)]
                Score_list.append(temp)
            

            score_array = np.array(Score_list)
            score_array_sum = np.sum(score_array, axis=0)

            # Add of the normalized sum squared error for each of the nurtient for each food item in the basket
            # this way we can figure out which food item is contributing the most/least nutrients. It will help deciding which food item to knock off the basket
            for i in range(len(Score_list)):
                score_list_sum.append(sum(Score_list[i]))
            

           
            Score_total = calcScore(target, sum_C)
            currentScore = sum(Score_total)
            print("Basket before drop")
            print(basket)
            if ((currentScore > minScore) and (len(basket.index) > NoBasketEnteries)):
                print(basket)

                if((len(basket.index) > NoBasketEnteries)):
                    #drop the top #no_to_drop entries which are having large errors. This will retain the basket size to NoBasketEnteries
                    no_to_drop = len(basket.index) - NoBasketEnteries
                    print(f"No.to drop : {no_to_drop}")
                    top_list = sorted(range(len(score_list_sum)), key=lambda i: score_list_sum[i])[-no_to_drop:]
                    print(top_list)
                    basket.drop(index=top_list, inplace=True)
                    basket.reset_index(inplace = True, drop=True)
                    basket_NDB.drop(index=top_list, inplace=True)
                    basket_NDB.reset_index(inplace = True, drop=True)
                else:
            
                    maxpos = score_list_sum.index(max(score_list_sum)) 
                    basket.drop(index=maxpos, inplace=True)
                    basket.reset_index(inplace = True, drop=True)
                    basket_NDB.drop(index=maxpos, inplace=True)
                    basket_NDB.reset_index(inplace = True, drop=True)
              
            elif (currentScore <= minScore):
                continue
            print("Basket after drop")
            print(basket)
        else:
            print(f"Minimum score : {minScore}")
            print(basket_NDB)            
            break
    return basket_NDB
    