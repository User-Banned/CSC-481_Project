import numpy as np
import csv

# Class
priceRupee = []
priceUSD = []

# Laptop Model and Identifiers
model = []

# Features
rating = []
brand = []
processorBrand = []
processorTier = []
coreCount = []
threadCount = []
ramSize = []
primarySortageType = []
primaryStorageSize = []
secondaryStorageType = []
secondaryStorageSize = []
gpuBrand = []
gpuType = []
isTouch = []
displaySize = []
resolutionWidth = []
resolutionHeight = []
operatingSystem = []
warrentyPeriod = []

# Feature Gatering from CSV
with open('laptops.csv', mode ='r')as file:
    csvFile = csv.DictReader(file)
    for lines in csvFile:
        priceRupee.append(lines['Price'])
        rating.append(lines['Rating'])
        brand.append(lines['brand'])
        model.append(lines['Model'])
        processorBrand.append(lines['processor_brand'])
        processorTier.append(lines['processor_tier'])
        coreCount.append(lines['num_cores'])
        threadCount.append(lines['num_threads'])
        ramSize.append(lines['ram_memory'])
        primarySortageType.append(lines['primary_storage_type'])
        primaryStorageSize.append(lines['primary_storage_capacity'])
        secondaryStorageType.append(lines['secondary_storage_type'])
        secondaryStorageSize.append(lines['secondary_storage_capacity'])
        gpuBrand.append(lines['gpu_brand'])
        gpuType.append(lines['gpu_type'])
        isTouch.append(lines['is_touch_screen'])
        displaySize.append(lines['display_size'])
        resolutionWidth.append(lines['resolution_width'])
        resolutionHeight.append(lines['resolution_height'])
        operatingSystem.append(lines['OS'])
        warrentyPeriod.append(lines['year_of_warranty'])

# Make 2D Array Dataset | in form of [feature][index value]
laptopDataset = [brand,processorBrand,processorTier,coreCount,
                 threadCount,ramSize,primarySortageType,primaryStorageSize,
                 secondaryStorageType,secondaryStorageSize,gpuBrand,
                 gpuType,isTouch,displaySize,resolutionWidth,
                 resolutionHeight,operatingSystem,warrentyPeriod]

# Turn Rupee into USD
for price in priceRupee:
    priceUSD.append(round((int(price)/83.36)*100)/100)

# Definitions
def linReg():
    return

print(priceUSD)
print(np.max(priceUSD))
