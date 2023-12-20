# data directory

Training data should be placed under this directory.
## data Organization

```data/
    ├── images/				<- resized all images
    │	├──D1BN_TM10.5_0000_G001_1.png
    │	├──D1BN_TM10.5_0002_L003_1.png
    │	...
    │ 	└──G2AN_TM10.5_0415_L036_8.png
    │
    ├── labels/				<- label images
    │	├──D1BN_TM10.5_0010_G001_2.png
    │	├──D1BN_TM10.5_0012_L003_2.png
    │	...
    │ 	└──G2AN_TM11.5_0347_L038_1.png
    │
    ├──ImageDataTable.csv  			<- Images type table (Name,GenType,TM,Pos,GL,X_size,Y_size,NormPos)
    ├──LabelDataTable.csv			<- Labels type table (Name,GenType,TM,Pos,GL,X_size,Y_size,NormPos)
    └── README.md          			<- The top-level README for developers.

```
