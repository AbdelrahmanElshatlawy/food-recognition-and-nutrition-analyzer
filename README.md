# food-recognition-and-nutrition-analyzer
End-to-end pipeline that estimates nutrition from a single meal photo. It detects foods with YOLOv8, builds masks (segmentation or GrabCut), estimates volume → weight using MiDaS depth, then converts weight to calories, protein, carbs, fat from a CSV. Outputs an annotated image and per-item/total CSVs.
