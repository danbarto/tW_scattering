import awkward1 as ak
import numpy as np

class Cutflow:
    
    def __init__(self, output, ev, weight, selection=None ):
        '''
        output: the accumulator object
        ev: NanoEvent
        weight: coffea analysis_tools Weights object
        '''
        self.ev = ev
        self.weight = weight.weight()
        self.output = output
        self.selection = None
        self.addRow('entry', (ak.ones_like(self.weight)==1) )
        
        
    def addRow(self, name, selection, cumulative=True):
        '''
        If cumulative is set to False, the cut will not be added to self.selection
        '''
        if self.selection is None and selection is not None:
            self.selection = selection
        elif selection is not None and cumulative == False:
            selection = self.selection & selection
        elif selection is not None:
            self.selection = self.selection & selection
            selection = self.selection
        
            
        
        if selection is not None:
            self.output[self.ev.metadata['dataset']][name] += sum(self.weight[ selection ] )
            self.output[self.ev.metadata['dataset']][name+'_w2'] += sum(self.weight[ selection ]**2) 
        else:
            self.output[self.ev.metadata['dataset']][name] += sum(self.weight)
            self.output[self.ev.metadata['dataset']][name+'_w2'] += sum(self.weight**2)
  
        
