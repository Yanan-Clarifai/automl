import enum
import pickle
import yaml


class ConceptEnum(enum.Enum):
    """
    Define your enum here:
        Example:
            dog = 0
            cat = 1
    
    To enforce bijection
    
        @enum.unique
        class YourEnum
            dog = 0
            cat = 1
    """
    
    @classmethod
    def assert_bijective(cls):
        enum.unique(cls)
    
    @classmethod
    def to_indices(cls, concept_list, sort_index=True):
        ind = []
        for name in concept_list:
            ind.append(cls[name].value)

        if sorted:
            ind.sort()

        return ind
    
    @classmethod
    def to_concepts(cls, index_list, sort_index=True):        
        if sorted:
            index_list.sort()

        con = []
        for idx in index_list:
            con.append(cls(idx).name)
        
        return con
    
    @classmethod
    def save(cls, filename):
        with open(filename, 'wb') as f:
            pickle.dump(file=f, obj=cls)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            out = pickle.load(file=f)
        return out

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as f:
            out = yaml.load(f)
        return cls(out['name'], out['concepts'], start=0)
    
    @classmethod
    def from_dict(cls, dictionary):
        return cls(dictionary['name'], dictionary['concepts'], start=0)
    
    @classmethod
    def to_yaml(cls, filename):
        with open(filename, 'w') as f:
            out = {'name': cls.__name__, 'concepts': cls._member_names_}
            f.write(yaml.dump(out))
    
    @classmethod
    def describe(cls):
        print(cls.__name__)
        print(cls._member_map_)