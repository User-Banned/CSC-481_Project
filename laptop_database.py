import csv

# Feature Gathering from CSV
def get_features_from_csv(filename):
    brand = []
    processorBrand = []
    processorTier = []
    primaryStorageType = []
    secondaryStorageType = []
    gpuBrand = []
    gpuType = []
    isTouch = []
    operatingSystem = []
    model = []
    priceUSD = []
    rating = []
    coreCount = []
    threadCount = []
    ramSize = []
    primaryStorageSize = []
    secondaryStorageSize = []
    displaySize = []
    resolutionWidth = []
    resolutionHeight = []
    warrentyPeriod = []

    with open(filename, mode='r') as file:
        csvFile = csv.DictReader(file)
        for line in csvFile:
            model.append(line['Model'])
            brand.append(line['brand'])
            processorBrand.append(line['processor_brand'])
            processorTier.append(line['processor_tier'])
            primaryStorageType.append(line['primary_storage_type'])
            secondaryStorageType.append(line['secondary_storage_type'])
            gpuBrand.append(line['gpu_brand'])
            gpuType.append(line['gpu_type'])
            isTouch.append(line['is_touch_screen'])
            operatingSystem.append(line['OS'])
            priceUSD.append(round((int(line['Price']) / 83.36) * 100) / 100) # Price Conversion from Rupee to USD
            rating.append(int(line['Rating']))
            coreCount.append(int(line['num_cores']))
            threadCount.append(int(line['num_threads']))
            ramSize.append(int(line['ram_memory']))
            primaryStorageSize.append(int(line['primary_storage_capacity']))
            secondaryStorageSize.append(int(line['secondary_storage_capacity']))
            displaySize.append(float(line['display_size']))
            resolutionWidth.append(int(line['resolution_width']))
            resolutionHeight.append(int(line['resolution_height']))
            if line['year_of_warranty'] == 'No information':
                warrentyPeriod.append(0)
            else:
                warrentyPeriod.append(int(line['year_of_warranty']))

    return model, brand, processorBrand, processorTier, primaryStorageType, secondaryStorageType, gpuBrand, gpuType, \
           isTouch, operatingSystem, priceUSD, rating, coreCount, threadCount, ramSize, primaryStorageSize, \
           secondaryStorageSize, displaySize, resolutionWidth, resolutionHeight, warrentyPeriod