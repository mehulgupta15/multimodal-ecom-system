import os
import csv
import requests

def fetch_production_catalog():
    print("=== Starting Day 6 Data Ingestion Pipeline ===")
    
    # 1. Define local directory paths on your E: drive
    csv_path = "./data/products_catalog.csv"
    img_dir = "./data/data_images"
    
    os.makedirs(img_dir, exist_ok=True)
    
    # 2. Query the public e-commerce dummy API endpoint
    api_url = "https://fakestoreapi.com/products"
    print("Connecting to live product server...")
    
    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        products = response.json()
    except Exception as e:
        print(f"❌ Network connection failed: {e}")
        return

    print(f"Successfully retrieved {len(products)} product records! Processing downloads...")
    
    # 3. Open the CSV spreadsheet to write data
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write our master schema columns
        writer.writerow(["product_id", "title", "category", "image_path", "description"])
        
        # 4. Loop through every scraped product item
        for item in products:
            p_id = f"prod_{item['id']}"
            title = item['title']
            category = item['category']
            desc = item['description']
            img_url = item['image']
            
            # Formulate the safe local image filename
            clean_filename = f"{p_id}.jpg"
            local_img_path = os.path.join(img_dir, clean_filename)
            
            # Download the physical product image binary file
            try:
                img_data = requests.get(img_url, timeout=30).content
                with open(local_img_path, "wb") as f:
                    f.write(img_data)
                print(f" Saved: {local_img_path}")
            except Exception as e:
                print(f"⚠️ Failed to download image for {p_id}: {e}")
                continue
                
            # Write the completed row metadata to the spreadsheet ledger
            writer.writerow([p_id, title, category, local_img_path, desc])
            
    print("\n🎉 Data ingestion complete! Spreadsheet populated and images saved.")
    print(f"Look inside your '{img_dir}' folder to see the real data.")

if __name__ == "__main__":
    fetch_production_catalog()