import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import io

def run_occlusion_analysis(img, model, class_names, top_k=3, patch_size=20, stride=10):
    img_resized = img.resize(model.input_shape[1:3])
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    preds = model.predict(img_array, verbose=0)[0]
    top_k_indices = preds.argsort()[-top_k:][::-1]
    
    results = []
    
    for class_idx in top_k_indices:
        heatmap = np.zeros(model.input_shape[1:3])
        original_pred = preds[class_idx]
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        
        for y in range(0, model.input_shape[1]-patch_size+1, stride):
            for x in range(0, model.input_shape[2]-patch_size+1, stride):
                temp_img = img_array.copy()
                temp_img[0, y:y+patch_size, x:x+patch_size, :] = 0
                occluded_pred = model.predict(temp_img, verbose=0)[0][class_idx]
                heatmap[y:y+patch_size, x:x+patch_size] = original_pred - occluded_pred
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_resized)
        ax1.set_title(f"{class_name}\nScore: {original_pred:.4f}")
        ax1.axis('off')
        
        ax2.imshow(img_resized)
        ax2.imshow(heatmap, cmap='jet', alpha=0.5)
        ax2.set_title(f'Occlusion Map: {class_name}')
        ax2.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        results.append({
            'class_name': class_name,
            'score': float(original_pred),
            'image_bytes': buf.read()
        })
    
    return results
