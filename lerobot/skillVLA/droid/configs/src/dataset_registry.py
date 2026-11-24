"""Named raw/converted/training dataset families shared by DROID workflows."""
from pathlib import Path


def configure(config):
    config = dict(config)
    selected = config.get('droid_dataset')
    if isinstance(selected, dict):
        if not selected.get('name') or 'group' not in selected:
            raise ValueError('droid_dataset requires name and group')
        name = selected['name']
        registry = selected.get('sources', {})
        config['droid_dataset_group'] = selected['group']
    else:
        # Read old frozen scalar-name + registry configs without changing them.
        name = selected
        registry = config.get('droid_datasets', {})
    if name is not None:  # Old frozen configs remain self-contained.
        if not isinstance(name, str) or not isinstance(registry, dict) or name not in registry:
            raise ValueError(f'Unknown DROID dataset: {name}')
        item = registry[name]
        required = {'raw_root', 'calibration_root', 'converted_dataset', 'training_dataset'}
        if not isinstance(item, dict) or required - set(item):
            raise ValueError(f'DROID dataset {name} requires {sorted(required)}')
        for key in ('converted_dataset', 'training_dataset'):
            value = item[key]
            if not isinstance(value, str) or Path(value).name != value or value in ('', '.', '..'):
                raise ValueError(f'{key} must be a single dataset folder name')
        for key in ('raw_root', 'calibration_root'):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError(f'DROID dataset {name}: {key} must be a nonempty path')
        for key in ('raw_root', 'calibration_root', 'annotations_path'):
            config.setdefault(key, item.get(key, ''))
        config.setdefault('source_dataset', item['converted_dataset'])
        config.setdefault('target_dataset', item['training_dataset'])
    # Expand dataset placeholders; checkpoint/horizon choices stay explicit.
    for key in ('dp_run_name', 'skillset_seg_name', 'output_name'):
        if isinstance(config.get(key), str):
            for field in ('target_dataset', 'source_dataset'):
                placeholder = '{' + field + '}'
                if placeholder in config[key]:
                    if field not in config:
                        raise ValueError(f'{key} requires {field}')
                    config[key] = config[key].replace(placeholder, config[field])
            if '{droid_group_suffix}' in config[key]:
                import runpy
                groups = runpy.run_path(str(Path(__file__).with_name('dataset_groups.py')))
                config[key] = config[key].replace('{droid_group_suffix}', groups['suffix'](config))
    return config
