from src.data_processing.processing import ProcessorRow

# TODO: batch evaluation
class ApplyModel(ProcessorRow):
    
    def __init__(self, model):
        self.model = model

    def apply_to_row(self, row_original):
        row = row_original.copy()
        img = row.pixel_array
        row['output'] = [self.model(img)]

        return row

class ApplyMeasure(ProcessorRow):
    
    def __init__(self, measure, name):
        self.measure = measure
        self.name = name

    def apply_to_row(self, row_original):
        row = row_original.copy()
        output = row.output
        if 'target' in row.keys():
            target = row.target
            row[self.name] = self.measure(output, target)
        else:
            row[self.name] = self.measure(output)
        return row