# Data directory

Training data should be placed under this directory.
## Data Organization

```data/
    ├── images/				        <- resized all images
    │	├──D1BN_TM10.5_0000_G001_1.png
    │	├──D1BN_TM10.5_0002_L003_1.png
    │	...
    │ 	└──G2AN_TM10.5_0415_L036_8.png
    │
    ├── labels/				        <- label images
    │	├──D1BN_TM10.5_0010_G001_2.png
    │	├──D1BN_TM10.5_0012_L003_2.png
    │	...
    │ 	└──G2AN_TM11.5_0347_L038_1.png
    │
    ├──contours
    │	├──D1BN_TM10.5_0000_G001_1.png.csv
    │	├──D1BN_TM10.5_0002_L003_1.png.csv
    │	...
    │   └──G2AN_TM10.5_0415_L036_8.png.csv
    │
    ├──ImageDataTable.csv  			<- Images type table (Name,GenType,TM,Pos,GL,X_size,Y_size,NormPos)
    ├──LabelDataTable.csv			<- Labels type table (Name,GenType,TM,Pos,GL,X_size,Y_size,NormPos)
    ├──create_h5_split_class_id.py  <- Cahnge Prediction Images (.png) to BD5 (.h5)
    ├──resize.py                    <- Resize Prediction images
    └── README.md          			<- The top-level README for developers.

```
