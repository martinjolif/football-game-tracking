if __name__ == "__main__":
    from image_api import call_image_apis

    image_path = "../player_detection/data/yolov8-format/test/images/08fd33_3_6_png.rf.fc5f6b621352712ee4e8fcdb56074c77.jpg"
    results = call_image_apis(image_path)
    for endpoint, result in results.items():
        print(f"Results from {endpoint}:")
        print(result)