import os
import requests
import cv2
from configparser import ConfigParser
from typing import Dict, List
from activitygen.merge.element import Element
from activitygen.merge.merger import show_elements
import pandas as pd

# A table stores elements that are contained in the same table
# and column


class TableDetector:

    def __init__(self, output_root):
        self.output_root = output_root
        # Table detection
        self.TABLE_API_URL = "https://api-inference.huggingface.co/models/TahaDouaji/detr-doc-table-detection"
        # Table structure recognition
        self.STRUCTURE_API_URL = "https://api-inference.huggingface.co/models/microsoft/table-transformer-structure-recognition"

        config_parser = ConfigParser()
        config_parser.read("configs/api_keys.cfg")
        self.api_key = config_parser.get("MAIN", "HUGGINGFACE_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-wait-for-model": "true",
        }

    def query(self, img_path, url):
        with open(img_path, "rb") as f:
            data = f.read()
        response = requests.post(url, headers=self.headers, data=data)
        return response.json()

    def detect(self, img_path, elements: Dict[int, Element], threshold=0.9, show=False):
        img_name = img_path.split("/")[-1].split(".")[0]
        og_image = cv2.imread(img_path)

        table_detections = self.query(img_path, self.TABLE_API_URL)

        tables = {}

        for index, table_data in enumerate(table_detections):
            if table_data["score"] < threshold:  # exclude tables with low confidence
                continue
            # Get bounding box for every table in output
            bbox_table = table_data["box"]

            # Structure recognition woks better with padding
            padding = 50
            y1 = max(0, bbox_table["ymin"] - padding)
            y2 = min(og_image.shape[0], bbox_table["ymax"] + padding)
            x1 = max(0, bbox_table["xmin"] - padding)
            x2 = min(og_image.shape[1], bbox_table["xmax"] + padding)

            # Crop the table from the original image
            padded_image = og_image[y1:y2, x1:x2].copy()

            # Store table image in folder with table name and index
            table_path = self.output_root + img_name + f"_{index}_padded.png"
            cv2.imwrite(table_path, padded_image)

            table_structure = self.query(table_path, self.STRUCTURE_API_URL)
            tables[index] = {
                "id": len(tables),
                "columns": [],
                "rows": [],
                "x1": bbox_table["xmin"],
                "y1": bbox_table["ymin"],
                "x2": bbox_table["xmax"],
                "y2": bbox_table["ymax"],
            }
            table = tables[index]
            # Visualize the table structure with cv2 rectangle and put label
            for area in table_structure:
                bbox_area = area["box"]

                # convert relative coordinates to absolute
                xmin = bbox_area["xmin"] + x1
                ymin = bbox_area["ymin"] + y1
                xmax = bbox_area["xmax"] + x1
                ymax = bbox_area["ymax"] + y1
                area_element = Element(
                    -1,
                    (xmin, ymin, xmax, ymax),
                    area["label"],
                )
                match area["label"]:
                    case "table column":
                        table["columns"].append(area_element)
                    case "table row" | "table column header":
                        table["rows"].append(area_element)

                # Visualize and save
                cv2.rectangle(
                    padded_image,
                    (bbox_area["xmin"], bbox_area["ymin"]),
                    (bbox_area["xmax"], bbox_area["ymax"]),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    padded_image,
                    area["label"],
                    (bbox_area["xmin"], bbox_area["ymin"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            table_structure_path = (
                self.output_root + img_name + f"_{index}_structure.png"
            )
            cv2.imwrite(table_structure_path, padded_image)

            # Sort rows and columns
            table["rows"].sort(key=lambda row: row.y1)
            table["columns"].sort(key=lambda column: column.x1)

            # Assign elements to cells
            self.assign_elements(elements, table)

            # postprocess table and elements
            keep_table = self.postprocess(elements, table)
            if not keep_table:
                del tables[index]
                # delete file specified by table_path variable
                # os.remove(table_path)
                break

            if show:
                cv2.imshow("Table", padded_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # visualize tables and elements
        table_structures = [row for table in tables.values() for row in table["rows"]]
        table_structures += [
            column for table in tables.values() for column in table["columns"]
        ]

        visualization = show_elements(
            og_image,
            table_structures + list(elements.values()),
            show=show,
            win_name="Tables and elements",
            wait_key=0,
        )

        cv2.imwrite(self.output_root + img_name + "_POSTPROCESSED.png", visualization)

        return tables

    # deletes tables that contain buttons or blocks and deletes non-text elements that are contained in table
    def postprocess(self, elements: Dict[int, Element], table: Dict):
        table_element = Element(
            table["id"],
            (table["x1"], table["y1"], table["x2"], table["y2"]),
            "table",
        )

        # check if elements intersect with table
        keep_table = True
        delete_elements = []
        for element_id, element in elements.items():
            if element.cell is not None:  # element is in table
                if element.category == "Button":  # button in table
                    column = element.cell[1]
                    # if no more buttons in the same column -> delete table
                    if not any(
                        [
                            (el.category == "Button" and el.id != element.id)
                            for elements in table["df"][column]
                            for el in elements
                        ]
                    ):
                        keep_table = False
                        break

                # if something different -> delete this element (mainly EditText)
                if element.category != "Text":
                    delete_elements.append(element_id)

        # delete components contained in table
        if keep_table:
            for id in delete_elements:
                # update parents / children
                for child_id in elements[id].children:
                    elements[child_id].parent_id = None
                if elements[id].parent_id is not None:
                    elements[elements[id].parent_id].children.remove(id)

            for id in delete_elements:
                del elements[id]
        return keep_table

    # Assigns elements to a cell of a pandas df
    def assign_elements(self, elements: Dict[int, Element], table: Dict):
        df = pd.DataFrame(
            index=range(len(table["rows"])), columns=range(len(table["columns"]))
        )
        df = df.map(lambda x: [])  # init cells with empty lists
        for element in elements.values():
            best_row = -1
            best_ia = 0
            for row_id, row in enumerate(table["rows"]):
                ia = element.calc_intersection_area(row)[0]
                if ia > best_ia:
                    best_row = row_id
                    best_ia = ia
            best_column = -1
            best_ia = 0
            for column_id, column in enumerate(table["columns"]):
                ia = element.calc_intersection_area(column)[0]
                if ia > best_ia:
                    best_column = column_id
                    best_ia = ia

            if best_row != -1 and best_column != -1:
                df.iloc[best_row, best_column].append(element)
                element.cell = (best_row, best_column)

        # Merge columns with empty header to the left
        col = 1  # skip first column
        while col < len(df.columns):
            # Check if the first row entry is an empty list -> no column title
            if df.iloc[0, col] == []:
                # Merge lists of left column with lists of removed column
                df.iloc[:,col - 1] = df.iloc[:, col - 1] + df.iloc[:,col]

                # Update location of affected elements
                for elements_col in df.iloc[:,col]:
                    for element in elements_col:
                        element.cell = (element.cell[0], df.columns[col - 1])

                df.drop(df.columns[col], axis=1, inplace=True)

                # Update column area
                table["columns"][col - 1].x2 = table["columns"][col].x2
                table["columns"][col - 1].y2 = table["columns"][col].y2
                table["columns"].pop(col)
            else:
                col += 1

        table["df"] = df
