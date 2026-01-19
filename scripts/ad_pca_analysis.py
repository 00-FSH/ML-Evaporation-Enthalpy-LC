import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os
import matplotlib

# ================================
# 1. Load data
# ================================
train_file = r'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\数据集主成分分析.xlsx'
candidate_file = r'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\候选材料主成分分析.xlsx'

train_df = pd.read_excel(train_file)
cand_df = pd.read_excel(candidate_file)

descriptor_cols = train_df.columns.tolist()

# ================================
# 2. Handle missing values
# ================================
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(train_df[descriptor_cols])
X_cand_imputed = imputer.transform(cand_df[descriptor_cols])

# ================================
# 3. Standardization
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_imputed)
X_cand = scaler.transform(X_cand_imputed)

# ================================
# 4. PCA
# ================================
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_cand_pca = pca.transform(X_cand)

print("PC1 explained variance:", pca.explained_variance_ratio_[0])
print("PC2 explained variance:", pca.explained_variance_ratio_[1])

# ================================
# 5. AD Analysis (Euclidean distance)
# ================================
centroid = X_train.mean(axis=0)

def euclidean_distance(X, center):
    return np.linalg.norm(X - center, axis=1)

train_dist = euclidean_distance(X_train, centroid)
cand_dist = euclidean_distance(X_cand, centroid)

threshold = train_dist.mean() + 2 * train_dist.std()
print(f"AD threshold (mean + 2σ): {threshold:.3f}")

cand_AD_status = [
    'Inside AD' if d <= threshold else 'Outside AD'
    for d in cand_dist
]

# ================================
# 6. Prepare output directory
# ================================
output_dir = r'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ================================
# 7. Save AD results
# ================================
cand_results = cand_df.copy()
cand_results['Distance_to_Centroid'] = cand_dist
cand_results['AD_Status'] = cand_AD_status
cand_results.to_excel(os.path.join(output_dir, 'candidate_AD_results.xlsx'), index=False)

# ================================
# 8. Save PCA coordinates for Origin
# ================================
train_pca_df = pd.DataFrame({
    'PC1': X_train_pca[:, 0],
    'PC2': X_train_pca[:, 1],
    'Set': 'Training',
    'AD_Status': 'Training'
})

cand_pca_df = pd.DataFrame({
    'PC1': X_cand_pca[:, 0],
    'PC2': X_cand_pca[:, 1],
    'Set': 'Candidate',
    'AD_Status': cand_AD_status
})

plot_df = pd.concat([train_pca_df, cand_pca_df], ignore_index=True)
plot_df.to_excel(os.path.join(output_dir, 'PCA_AD_coordinates_for_Origin.xlsx'), index=False)

# Save explained variance
variance_df = pd.DataFrame({
    'Principal_Component': ['PC1', 'PC2'],
    'Explained_Variance_Ratio': pca.explained_variance_ratio_
})
variance_df.to_excel(os.path.join(output_dir, 'PCA_explained_variance.xlsx'), index=False)

print("All output files saved to:", output_dir)

# ================================
# 9. Plot PCA with AD ellipses (final version)
# ================================

# Global font
matplotlib.rcParams['font.family'] = 'Times New Roman'

# Functions for ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

# Color mapping (RGB normalized)
color_map = {
    'Training': (150/255, 210/255, 176/255),           # 浅绿色
    'Candidate_Inside AD': (38/255, 129/255, 182/255), # 蓝色
    'Candidate_Outside AD': (1, 0, 0)                  # 红色
}

plt.figure(figsize=(8.31, 7.40), dpi=1200)
ax = plt.gca()

# Training points
train_points = plot_df[plot_df['Set']=='Training'][['PC1','PC2']].values
plt.scatter(train_points[:,0], train_points[:,1], c=[color_map['Training']], marker='o', s=25, label='Training', alpha=1.0)
plot_point_cov(train_points, nstd=2, alpha=0.2, color=color_map['Training'], ax=ax)

# Candidate points inside AD
cand_inside = plot_df[(plot_df['Set']=='Candidate') & (plot_df['AD_Status']=='Inside AD')][['PC1','PC2']].values
plt.scatter(cand_inside[:,0], cand_inside[:,1], c=[color_map['Candidate_Inside AD']], marker='o', s=25, label='Candidate (Inside AD)', alpha=1.0)
plot_point_cov(cand_inside, nstd=2, alpha=0.2, color=color_map['Candidate_Inside AD'], ax=ax)

# Candidate points outside AD
cand_outside = plot_df[(plot_df['Set']=='Candidate') & (plot_df['AD_Status']=='Outside AD')][['PC1','PC2']].values
plt.scatter(cand_outside[:,0], cand_outside[:,1], c=[color_map['Candidate_Outside AD']], marker='o', s=25, label='Candidate (Outside AD)', alpha=1.0)
# 不绘制 AD 外椭圆

# Axes labels
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f} %)', fontsize=28, fontweight='bold', fontname='Times New Roman')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f} %)', fontsize=28, fontweight='bold', fontname='Times New Roman')

# Tick labels
plt.xticks(fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')

# Legend upper left, no border, bold, 28pt
leg = plt.legend(
    loc='upper left',
    frameon=False,
    prop={'family':'Times New Roman', 'weight':'bold', 'size':20}
)

plt.tight_layout()

# Save figure
plt.savefig(os.path.join(output_dir, 'PCA_AD_ellipse_final_RGB.tiff'), dpi=1200, format='tiff')
plt.show()
