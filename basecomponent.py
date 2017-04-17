class BaseComponent(object):
    '''
    Base class for all components. Host for common helpers instead
    of repeating in each subclass.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg['name']
        
        
    def execute(self, input_data, input_directory, output_directory):
        '''
        Every component should perform its primary operation - detection, recognition or
        file writing - in execute().
        
        Returns: a dict of the component's outputs, where keys are 'reports', 'annotatedimages', 
            'annotatedframes', etc.
            
            Each report should be like this
            [
                {'labels':[{'label':'cat', 'confidence':0.8}, {'label':'lion', 'confidence':0.3}], 'rect':[x1,y1,x2,y2] },
                {'labels':['dog','sheep'], 'rect':[x1,y1,x2,y2], 'confidence':0.8}
            ]
            confidence values are optional.
            The coordinates should always be full image coordinates even if input to the component was ROI output of another 
            component.
        '''
        
        pass
        
        
        
        
    def completed(self, input_data, input_directory, output_directory):
        '''
        Some components need to know when processing of input file
        has completed.
        '''
        pass
    
