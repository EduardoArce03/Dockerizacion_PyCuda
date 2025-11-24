

from filters.emboss_filter import EmbossFilter
from filters.blur_filter import BlurFilter
from filters.laplace_filter import LaplaceFilter


class FilterFactory:

    
    AVAILABLE_FILTERS = {
        'emboss': EmbossFilter,
        'blur': BlurFilter,
        'laplace': LaplaceFilter,
    }
    
    @classmethod
    def create_filter(cls, filter_name):

        filter_name = filter_name.lower().strip()
        
        if filter_name not in cls.AVAILABLE_FILTERS:
            available = ', '.join(cls.AVAILABLE_FILTERS.keys())
            raise ValueError(
                f"Filtro '{filter_name}' no encontrado. "
                f"Filtros disponibles: {available}"
            )
        
        filter_class = cls.AVAILABLE_FILTERS[filter_name]
        return filter_class()
    
    @classmethod
    def get_available_filters(cls):

        filters_info = []
        
        for name, filter_class in cls.AVAILABLE_FILTERS.items():
            try:
                temp_filter = filter_class()
                filters_info.append({
                    "name": name,
                    "description": temp_filter.description,
                    "parameters": temp_filter.get_parameters(),
                    "recommended_blocks": temp_filter.get_recommended_block_sizes()
                })
            except Exception as e:
                filters_info.append({
                    "name": name,
                    "description": f"Error al cargar filtro: {str(e)}",
                    "parameters": {},
                    "recommended_blocks": []
                })
        
        return filters_info
    
    @classmethod
    def validate_filter_parameters(cls, filter_name, parameters):

        try:
            filter_instance = cls.create_filter(filter_name)
            filter_params = filter_instance.get_parameters()
            
            for param_name, param_config in filter_params.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    param_type = param_config.get('type')
                    
                    if param_type == 'int' and not isinstance(value, int):
                        return False, f"Parámetro '{param_name}' debe ser entero"
                    
                    if param_type == 'float' and not isinstance(value, (int, float)):
                        return False, f"Parámetro '{param_name}' debe ser numérico"
                    
                    if 'min' in param_config and value < param_config['min']:
                        return False, f"Parámetro '{param_name}' debe ser >= {param_config['min']}"
                    
                    if 'max' in param_config and value > param_config['max']:
                        return False, f"Parámetro '{param_name}' debe ser <= {param_config['max']}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)


class FilterManager(FilterFactory):
    pass
