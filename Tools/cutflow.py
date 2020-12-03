

class Cutflow:
    
    def __init__(self, output, df, cfg, processes, selection=None, weight=None ):
        '''
        If weight=None a branch called 'weight' in the dataframe is assumed
        '''
        self.df = df
        if weight is not None:
            self.weight = weight
        else:
            self.weight = df['weight']
        self.lumi = cfg['lumi']
        self.cfg = cfg
        self.output = output
        self.processes = processes
        self.selection = None
        self.addRow('entry', selection)
        
        
        
    def addRow(self, name, selection, cumulative=True):
        '''
        If cumulative is set to False, the cut will not be added to self.selection
        '''
        if self.selection is None and selection is not None:
            self.selection = selection
        elif selection is not None and cumulative == False:
            selection = self.selection & selection
        elif selection is not None:
            self.selection &= selection
            selection = self.selection
            
        
        for process in self.processes:
            if selection is not None:
                self.output[process][name] += ( sum(self.weight[ (self.df['dataset']==process) & (selection) ].flatten() )*self.lumi )
                self.output[process][name+'_w2'] += ( sum((self.weight[ (self.df['dataset']==process) & selection ]**2).flatten() )*self.lumi**2 )
            else:
                self.output[process][name] += ( sum(self.weight[ (self.df['dataset']==process) ].flatten() )*self.lumi )
                self.output[process][name+'_w2'] += ( sum((self.weight[ (self.df['dataset']==process) ]**2).flatten() )*self.lumi**2 )
  
        
