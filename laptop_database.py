import csv

# Laptop Model
model = []

# Non-Numerical Features
brand = []
processorBrand = []
processorTier = []
primarySortageType = []
secondaryStorageType = []
gpuBrand = []
gpuType = []
isTouch = []
operatingSystem = []

# Numerical Features
priceUSD = []   # Note: Converted Price From Rupee
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

# Feature Gatering from CSV
def __getFeaturesFromCSV():
    with open('laptops.csv', mode ='r')as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            # Non-Numerical
            brand.append(lines['brand'])
            model.append(lines['Model'])
            processorBrand.append(lines['processor_brand'])
            processorTier.append(lines['processor_tier'])
            primarySortageType.append(lines['primary_storage_type'])
            secondaryStorageType.append(lines['secondary_storage_type'])
            gpuBrand.append(lines['gpu_brand'])
            gpuType.append(lines['gpu_type'])
            isTouch.append(lines['is_touch_screen'])
            operatingSystem.append(lines['OS'])

            # Neumerical
            priceUSD.append(round((int(lines['Price'])/83.36)*100)/100)
            rating.append(int(lines['Rating']))
            coreCount.append(int(lines['num_cores']))
            threadCount.append(int(lines['num_threads']))
            ramSize.append(int(lines['ram_memory']))
            primaryStorageSize.append(int(lines['primary_storage_capacity']))
            secondaryStorageSize.append(int(lines['secondary_storage_capacity']))
            displaySize.append(float(lines['display_size']))
            resolutionWidth.append(int(lines['resolution_width']))
            resolutionHeight.append(int(lines['resolution_height']))
            if lines['year_of_warranty']=='No information':
                warrentyPeriod.append(0)
            else:
                warrentyPeriod.append(int(lines['year_of_warranty']))
    return

__getFeaturesFromCSV()
