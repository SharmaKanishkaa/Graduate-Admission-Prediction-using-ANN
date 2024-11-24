# 📊 **Customer Segmentation Using Unsupervised Clustering**

This project focuses on **customer segmentation** using **unsupervised machine learning techniques**. The goal is to group customers into segments based on their features such as **age**, **income**, and **spending score** to help businesses target their marketing efforts more effectively.

---

## 🧑‍💻 **Dataset Overview**

The dataset contains customer information, including:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer (e.g., Male, Female).
- **Age**: Age of the customer.
- **Income**: Annual income of the customer.
- **Score**: Spending score, which is a metric representing the customer's spending behavior.

---

## 🔍 **Clustering Algorithms Used**

The primary focus of this project is on **unsupervised clustering** algorithms, which aim to group customers based on their similarities without prior labels. The following algorithms are applied:

### 1. **K-Means Clustering (From Scratch)**
   - **How It Works**:  
     K-Means divides the data into a predefined number of clusters (K). The algorithm iteratively assigns each data point to the nearest cluster center and updates the cluster centers until convergence.
   - **Steps Involved**:  
     - Initialize K centroids randomly.
     - Assign each data point to the nearest centroid.
     - Update centroids based on the mean of points assigned to the centroid.
     - Repeat the process until convergence.
   - **Pros**:  
     - Simple and easy to implement.
     - Efficient for large datasets.
   - **Cons**:  
     - Sensitive to the initial placement of centroids.
     - Requires the number of clusters (K) to be predefined.

### 2. **Agglomerative Hierarchical Clustering**
   - **How It Works**:  
     Agglomerative clustering is a bottom-up approach where each data point starts as its own cluster, and the algorithm iteratively merges the closest clusters until a stopping criterion (such as a specified number of clusters) is met.
   - **Pros**:  
     - Does not require the number of clusters to be predefined.
     - Suitable for smaller datasets.
   - **Cons**:  
     - Computationally expensive for large datasets.
     - Can be sensitive to noise and outliers.

### 3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
   - **How It Works**:  
     DBSCAN groups points that are closely packed together and marks points in low-density regions as outliers. It doesn’t require the number of clusters to be predefined, making it suitable for data with irregular shapes.
   - **Pros**:  
     - Identifies outliers effectively.
     - Does not require the number of clusters to be predefined.
   - **Cons**:  
     - Sensitive to the choice of parameters (epsilon and min_samples).
     - Struggles with clusters of varying density.

---

## 🧩 **Model Implementation and Results**

### **1. K-Means Clustering (From Scratch)**
   - **Objective**:  
     To implement the K-Means algorithm from scratch and segment customers based on their features: **age**, **income**, and **spending score**.
   - **Accuracy**:  
     Evaluated using **Silhouette Score** and **Inertia (within-cluster sum of squares)**. The K-Means algorithm creates clear and distinct clusters for the customers.

### **2. Agglomerative Hierarchical Clustering**
   - **Objective**:  
     To use a hierarchical approach to segment customers. The results are visualized using **dendrograms** to identify the optimal number of clusters.
   - **Performance**:  
     The clustering behavior is compared using **AgglomerativeClustering** from **scikit-learn** and visualized for interpretation.

### **3. DBSCAN**
   - **Objective**:  
     To apply **DBSCAN** and identify customer segments based on the density of data points. This method does not require the number of clusters to be predefined.
   - **Evaluation**:  
     **DBSCAN** successfully identified clusters and outliers, with parameter tuning for **epsilon** and **min_samples** to improve cluster quality.

---

## 📈 **Visualizations and Insights**

- **Elbow Method for K-Means**: The **Elbow method** was used to determine the optimal number of clusters for K-Means by plotting the **inertia** against the number of clusters and looking for the "elbow" point.
- **Dendrogram for Agglomerative Clustering**: The **dendrogram** helped in selecting the optimal number of clusters based on merging patterns.
- **DBSCAN Results**: Visualizations showed clusters of varying density, along with noise points that were classified as outliers.

---

## 📊 **Model Performance Summary**

| Algorithm                       | Description                                      | Result     |
|----------------------------------|--------------------------------------------------|------------|
| **K-Means Clustering**           | Segmenting customers into predefined clusters    | Clear clusters, Silhouette score and Inertia calculated |
| **Agglomerative Clustering**     | Hierarchical approach to clustering with dendrogram | Optimal number of clusters determined |
| **DBSCAN**                       | Density-based clustering, detecting outliers      | Clusters with noise/outliers identified |

---

## ⚙️ **Tools and Techniques**

- **Libraries Used:**  
  - `pandas`, `numpy` for data handling and manipulation.  
  - `matplotlib`, `seaborn` for data visualization.  
  - `scikit-learn` for clustering algorithms (K-Means, Agglomerative, DBSCAN).  
- **Clustering Techniques:**  
  - **K-Means**  
  - **Agglomerative Hierarchical Clustering**  
  - **DBSCAN**

---

## 🌟 **Key Takeaways**

- **Unsupervised clustering** can be effectively used for customer segmentation, helping businesses target specific customer groups for tailored marketing strategies.
- **K-Means** is a simple and efficient algorithm but requires the number of clusters to be predefined.
- **Agglomerative Clustering** and **DBSCAN** are useful when the number of clusters is not known in advance or when clusters have irregular shapes.
- **Visualizations** like the **Elbow Method**, **Dendrogram**, and **DBSCAN** plots help in interpreting clustering results and determining the optimal number of clusters.
