Search.setIndex({"docnames": ["BendingRadius", "basic_example", "code_docs", "examples", "index", "new_data_exporter", "new_data_loader", "phase_space_format", "supported_particles"], "filenames": ["BendingRadius.md", "basic_example.ipynb", "code_docs.rst", "examples.rst", "index.rst", "new_data_exporter.ipynb", "new_data_loader.ipynb", "phase_space_format.md", "supported_particles.md"], "titles": ["Express bending radius in kinetic energy", "Basic Example", "Code Documentation", "Examples", "ParticlePhaseSpace", "Writing a new data exporter", "Writing a new data loader", "Phase Space Format", "Supported particles"], "terms": {"The": [0, 1, 5, 6, 7, 8], "particl": [0, 1, 4, 5, 6, 7], "magnet": 0, "field": [0, 1, 2], "i": [0, 1, 2, 5, 6, 8], "given": [0, 2], "begin": 0, "equat": 0, "label": 0, "eqn": 0, "1": [0, 1, 2, 5, 6, 7, 8], "tag": 0, "r": [0, 1, 2, 6], "frac": 0, "p": [0, 1, 2, 5, 6, 7], "qb": 0, "gamma": [0, 1, 2, 8], "m_0": 0, "v": [0, 2], "beta": [0, 1, 2], "c": [0, 1, 2, 5, 6, 7, 8], "end": [0, 2], "meanwhil": [0, 1], "relatavist": [0, 2], "e_k": [0, 7], "mc": 0, "2": [0, 1, 2, 5, 6, 7, 8], "m_0c": 0, "sqrt": [0, 7], "our": [0, 2, 5, 6], "goal": [0, 6], "term": [0, 7], "firstli": 0, "we": [0, 1, 2, 5, 6, 7], "can": [0, 1, 2, 5, 6, 7], "us": [0, 1, 2, 5, 6, 7, 8], "momentum": [0, 1, 2, 4, 5, 6], "relat": 0, "e": [0, 2, 5, 6, 7], "2c": 0, "4": [0, 1, 5, 6], "ek": [0, 5, 6, 7], "e0": 0, "e_0": [0, 7], "2e_k": 0, "now": [0, 1, 6], "assum": 0, "ev": 0, "have": [0, 1, 5, 6, 7], "just": [0, 1, 5, 6, 7], "calcul": [0, 1, 2, 7], "unit": [0, 4, 6], "thi": [0, 1, 2, 4, 5, 6, 7, 8], "directli": [0, 1], "would": [0, 2, 5, 6, 7], "convert": [0, 7], "si": 0, "kg": [0, 7], "m": [0, 2, 7], "": [0, 2, 5, 7], "p_": 0, "q": [0, 2, 7], "substitut": 0, "bc": 0, "note": [0, 1, 2, 4, 6], "formula": 0, "electron": [0, 1, 2, 5, 6, 7, 8], "volt": 0, "rearrang": 0, "slightli": 0, "At": [0, 1, 2], "point": 0, "ar": [0, 1, 2, 5, 6, 8], "still": 0, "defin": [0, 1, 7], "If": [0, 1, 2, 6], "instead": 0, "joul": 0, "bcq": 0, "which": [0, 1, 2, 6], "same": [0, 1, 2], "magdalena": 0, "answer": 0, "let": [0, 5, 6], "check": [0, 1, 5, 6], "all": [0, 1, 2, 5, 6, 7], "make": [0, 1, 6, 7], "sens": [0, 1], "some": [0, 1, 5, 6], "basic": [0, 2, 3, 4, 6], "import": [0, 2, 3, 4, 5, 6, 7], "numpi": [0, 5, 6], "np": [0, 2, 5, 6], "from": [0, 1, 2, 5, 6, 7], "scipi": 0, "constant": 0, "elementary_charg": 0, "b": 0, "t": [0, 2], "10e6": 0, "ek_si": 0, "j": 0, "0": [0, 1, 2, 5, 6, 7, 8], "511e6": 0, "e0_si": 0, "first": [0, 1, 2, 5, 6], "version": 0, "r_brendan": 0, "r_magdalena": 0, "print": [0, 2], "f": [0, 5, 6], "n": [0, 5, 6], "nbend": 0, "nenergi": 0, "1e6": 0, "2f": 0, "mev": [0, 1, 2, 5, 6, 7, 8], "ncharg": 0, "2e": 0, "nin": 0, "ni": 0, "brendan": 0, "3f": 0, "10": [0, 1, 2], "00": 0, "charg": [0, 7, 8], "60e": 0, "19": [0, 1, 8], "In": [0, 1, 2, 6, 7], "018": 0, "onli": [0, 1, 2, 5, 7], "other": [0, 2, 7], "thing": 0, "worth": 0, "here": [0, 1, 2], "light": 0, "ratio": 0, "rest": [0, 2, 7], "becom": 0, "small": 0, "high": 0, "second": 0, "ha": [0, 2, 6], "valu": [0, 1, 5, 6, 7], "so": [0, 1, 6, 7, 8], "ignor": 0, "write": [0, 1, 2, 3, 4, 7], "much": 0, "more": [0, 1, 2], "simpli": 0, "approx": 0, "r_approxim": 0, "approxim": [0, 2], "nwhich": 0, "wrong": 0, "100": [0, 2], "ab": 0, "017": 0, "75": 0, "step": 1, "analysi": [1, 2], "code": [1, 4, 6, 7, 8], "alwai": [1, 2], "handl": [1, 2, 6], "through": 1, "an": [1, 2, 5, 6, 7, 8], "instanc": [1, 2, 6, 7], "dataload": [1, 2, 5, 6, 7], "class": [1, 2, 5, 6], "work": [1, 5, 6], "topa": [1, 2, 5], "For": [1, 6, 7], "list": [1, 2, 5], "current": [1, 8], "avail": [1, 5, 6], "see": 1, "instruct": 1, "new": [1, 2, 3, 4], "pathlib": [1, 2, 5, 6], "path": [1, 2, 5, 6], "sy": [1, 5, 6], "append": [1, 5, 6], "necessari": 1, "when": 1, "librari": [1, 2, 4, 6], "instal": [1, 6], "particlephasespac": [1, 2, 5, 6, 8], "phasespac": [1, 2, 5, 6, 7], "test_data_loc": 1, "test": [1, 2, 3, 6, 7, 8], "test_data": [1, 2, 5, 6], "coll_phasespace_xang_0": [1, 2, 5], "00_yang_0": [1, 2, 5], "00_angular_error_0": [1, 2, 5], "phsp": [1, 2, 5], "absolut": [1, 2, 6], "ps_data": [1, 2, 5], "load_topasdata": [1, 2, 5], "onc": 1, "pass": [1, 5, 6, 7, 8], "perform": 1, "3": [1, 5, 6], "print_energy_stat": [1, 2], "energi": [1, 2, 5, 6], "stat": [1, 2], "total": 1, "number": [1, 2], "311489": 1, "uniqu": [1, 2, 7], "speci": [1, 2], "308280": 1, "mean": [1, 6], "91": 1, "median": 1, "20": 1, "spread": 1, "iqr": 1, "03": 1, "min": 1, "01": 1, "max": 1, "35": 1, "2853": 1, "16": 1, "39": 1, "95": 1, "02": 1, "9": 1, "44": 1, "356": 1, "positron": [1, 8], "66": 1, "08": 1, "8": 1, "46": 1, "you": [1, 2, 5, 6, 7], "consist": 1, "three": 1, "differ": [1, 2, 6, 7], "also": [1, 2, 5, 6], "gener": [1, 2, 4, 5, 6, 7], "plot": [1, 2, 6, 7], "posit": [1, 2, 7], "plot_energy_histogram": [1, 2, 5], "5": [1, 2, 5, 6], "plot_position_histogram": [1, 2], "6": [1, 5, 6, 7], "plot_particle_posit": [1, 2, 6], "beam_direct": [1, 2, 6], "z": [1, 2, 5, 6, 7], "abov": [1, 7, 8], "weight": [1, 2, 5, 6, 7], "color": [1, 2], "anoth": 1, "wai": 1, "visualis": 1, "plot_beam_intens": [1, 2], "method": [1, 2, 5, 6, 7], "produc": [1, 2], "imag": [1, 2], "accumul": [1, 2], "intens": [1, 2], "7": 1, "xlim": [1, 2], "ylim": [1, 2], "grid": [1, 2], "fals": [1, 2], "actual": [1, 5, 6], "lot": 1, "illumin": 1, "than": [1, 2], "scatter": [1, 2, 6], "wa": [1, 7], "score": 1, "exit": 1, "novel": 1, "type": [1, 2, 5, 6, 7], "x": [1, 2, 5, 6, 7], "rai": 1, "collim": 1, "been": 1, "shape": [1, 5, 6], "squar": 1, "expect": [1, 6], "randomli": [1, 2], "easili": 1, "add": [1, 2, 8], "electron_p": 1, "pdg": [1, 7], "11": [1, 2, 5, 8], "get": [1, 6], "multipl": 1, "gamma_p": 1, "positron_p": 1, "one": [1, 2, 7], "follow": [1, 5, 6, 7, 8], "where": [1, 2, 5, 6, 7], "remov": [1, 2], "no_gamma_p": 1, "togeth": 1, "original_p": 1, "howev": [1, 7], "cannot": 1, "ident": 1, "exist": [1, 2, 5, 6], "realli": 1, "want": [1, 2, 5, 6], "do": [1, 2, 5], "need": [1, 5, 6, 7], "updat": [1, 2, 7, 8], "particle_id": 1, "courant": 1, "snyder": 1, "commonli": 1, "describ": 1, "distribut": 1, "transvers": 1, "save": [1, 2], "12": 1, "print_twiss_paramet": [1, 2], "rm": [1, 2], "y": [1, 2, 5, 6, 7], "epsilon": 1, "525901": 1, "673361": 1, "alpha": [1, 2], "619585": 1, "525727": 1, "976938": 1, "488084": 1, "154160": 1, "134526": 1, "208669": 1, "357446": 1, "159332": 1, "009164": 1, "971102": 1, "634599": 1, "471535": 1, "435510": 1, "881200": 1, "178092": 1, "141023": 1, "283977": 1, "888975": 1, "672388": 1, "591913": 1, "095932": 1, "13": 1, "plot_transverse_trace_space_scatt": [1, 2], "show": [1, 2], "what": [1, 6], "sometim": 1, "call": [1, 2, 6], "trace": [1, 2], "versu": [1, 2], "diverg": 1, "px": [1, 2, 5, 6, 7], "pz": [1, 2, 5, 6, 7], "red": 1, "ellips": [1, 2], "repres": [1, 2, 7], "enclos": 1, "should": [1, 2, 6, 7], "enclod": 1, "37": 1, "beam": [1, 2], "As": [1, 5], "altern": [1, 2], "sum": 1, "17": 1, "plot_transverse_trace_space_intens": [1, 2], "plot_twiss_ellips": [1, 2], "true": [1, 2, 6], "again": 1, "quit": [1, 7], "fairli": 1, "evenli": 1, "dsitribut": 1, "between": 1, "mm": [1, 2, 5, 6, 7], "almost": 1, "sinc": [1, 5, 6], "research": 1, "purpos": [1, 4, 6], "mai": [1, 2, 5, 6, 7], "wish": 1, "chang": 1, "becaus": [1, 5, 6, 7], "simul": [1, 6], "coordin": 1, "system": 1, "veri": [1, 2, 5], "possibl": 1, "interact": [1, 2], "panda": [1, 2], "store": [1, 6], "your": [1, 5, 6], "own": 1, "risk": 1, "u": [1, 2], "valid": [1, 7], "ani": [1, 2, 6, 7], "oper": [1, 8], "edit": 1, "underl": 1, "manual": 1, "reset_phase_spac": [1, 2], "like": [1, 5, 6, 7], "invalid": 1, "previous": 1, "quantiti": [1, 2, 7], "reduc": [1, 2], "requir": [1, 2, 4, 5, 6], "column": [1, 2, 4, 5, 6], "18": 1, "zero": [1, 5, 6], "doubl": [1, 5, 6], "reset": 1, "previou": 1, "re": 1, "desir": [1, 2], "fill_kinetic_": [1, 2, 7], "similarli": 1, "stage": 1, "demonstr": [1, 6], "time": [1, 2, 5, 6, 7], "had": 1, "written": 1, "dataexport": [1, 2, 5], "topas_export": [1, 2], "output_loc": [1, 2], "output_nam": [1, 2], "test_export": 1, "file": [1, 2, 5, 6, 7], "success": 1, "lt": [1, 5], "0x7fc6fa6804f0": 1, "gt": [1, 5], "page": [2, 4], "sphinx": 2, "automat": [2, 6], "scan": 2, "sourc": [2, 7], "load_pandasdata": 2, "input_data": 2, "particle_typ": [2, 5, 6], "none": [2, 5, 6], "load": [2, 5, 6], "format": [2, 4, 5, 6], "intern": [2, 7], "extern": 2, "case": [2, 5, 7], "dedic": 2, "pd": 2, "demo_data": 2, "datafram": 2, "py": [2, 5, 6, 7, 8], "pdg_code": [2, 5, 6, 7, 8], "id": [2, 5, 6, 7], "load_tibaraydata": 2, "ascii": [2, 6], "tibarai": 2, "rxy": 2, "bx": 2, "By": 2, "bz": 2, "g": [2, 7], "nmacro": 2, "rmacro": 2, "data_loc": [2, 5, 6], "tibaray_test": 2, "dat": [2, 5, 6], "read": [2, 4, 5, 6], "both": [2, 6], "binari": 2, "present": [2, 7], "behind": 2, "scene": 2, "reli": 2, "topas2numpi": 2, "_particlephasespac": [2, 5, 6], "data_load": 2, "hold": 2, "allow": [2, 4, 5], "user": [2, 6], "utilis": 2, "common": 2, "It": 2, "accept": 2, "paramet": [2, 3], "_dataloadersbas": [2, 5, 6], "assess_density_versus_r": 2, "rval": 2, "verbos": 2, "assess": 2, "how": [2, 6], "mani": [2, 7], "radiu": 2, "linspac": 2, "21": 2, "screen": 2, "str": [2, 5], "option": [2, 7], "main": 2, "direct": 2, "travel": 2, "default": 2, "return": 2, "density_data": 2, "contain": [2, 6], "roi": 2, "val": 2, "proport": 2, "insid": [2, 7], "calculate_twiss_paramet": 2, "twiss": [2, 3], "fill_beta_and_gamma": [2, 7], "factor": [2, 6, 7], "self": [2, 5, 6], "_ps_data": 2, "fill_direction_cosin": [2, 7], "cosin": 2, "respect": 2, "kinet": [2, 7], "fill_relativistic_mass": [2, 7], "relativist": [2, 7], "mass": [2, 7], "fill_rest_mass": [2, 7], "fill_veloc": [2, 7], "veloc": [2, 7], "filter_by_tim": 2, "t_start": 2, "t_finish": 2, "inclus": 2, "specfi": 2, "float": 2, "particleswith": 2, "new_inst": 2, "object": [2, 3, 7], "filter": 2, "get_downsampled_phase_spac": 2, "downsample_factor": 2, "randomlt": 2, "sampl": [2, 6], "larger": 2, "size": 2, "origin": [2, 5], "shuffl": 2, "befor": 2, "being": 2, "int": 2, "downsampl": 2, "normal": 2, "bin": 2, "vmin": 2, "vmax": 2, "weight_position_plot": [2, 6], "rather": 2, "everi": 2, "form": 2, "assign": 2, "over": 2, "2d": 2, "faster": 2, "gaussian": 2, "kde": 2, "addit": [2, 6], "well": 2, "set": [2, 7], "bool": 2, "overlaid": 2, "either": [2, 8], "turn": 2, "off": 2, "displai": 2, "rang": 2, "n_pixel": 2, "minimum": 2, "maximum": 2, "n_bin": 2, "titl": 2, "histogram": 2, "paritcl": 2, "each": [2, 7], "spci": 2, "plot_n_particles_v_tim": 2, "quickli": 2, "seper": [2, 3], "out": [2, 6], "bunch": 2, "appli": 2, "inform": 2, "slow": [2, 6], "could": [2, 7], "try": 2, "control": 2, "transpar": 2, "definit": [2, 8], "overlai": 2, "onto": 2, "file_nam": 2, "summari": 2, "json": 2, "specifi": [2, 5, 6], "thei": [2, 6], "filenam": 2, "directori": 2, "project_particl": 2, "distanc": [2, 7], "in_plac": 2, "project": 2, "forward": [2, 6], "back": 2, "serv": [2, 4], "crude": 2, "advanc": 2, "transport": 2, "up": 2, "absenc": 2, "forc": 2, "far": 2, "its": 2, "delet": 2, "deriv": [2, 7], "whenev": 2, "memori": 2, "footprint": 2, "phasespaceinst": 2, "output": 2, "featur": [2, 5, 6], "flag": 2, "histori": 2, "data": [3, 4, 7], "analyt": 3, "ad": [3, 4], "subtract": 3, "phase": [3, 4, 5, 6], "space": [3, 4, 5, 6], "manipul": [3, 4], "export": [3, 4, 6], "loader": [3, 4, 5], "aim": 4, "python": 4, "analys": 4, "exampl": [4, 5, 6], "support": [4, 7], "document": [4, 7, 8], "index": 4, "modul": 4, "search": 4, "similar": 5, "process": 5, "sai": 5, "he": 5, "sent": 5, "hi": 5, "me": 5, "To": [5, 7], "creat": 5, "_dataexportersbas": 5, "__particle_config__": [5, 6, 8], "particle_cfg": [5, 6], "newdataexport": 5, "def": [5, 6], "_define_required_column": 5, "_export_data": 5, "job": [5, 6], "fill": [5, 6, 7], "two": [5, 6, 7], "blank": 5, "must": [5, 6, 7], "These": [5, 7], "name": [5, 6, 7, 8], "match": 5, "specif": 5, "happen": 5, "hand": 5, "look": [5, 6], "_required_column": 5, "_output_nam": 5, "suffix": [5, 6], "els": 5, "writefilepath": 5, "_output_loc": 5, "header": [5, 6], "ty": [5, 6], "tz": [5, 6], "tpx": [5, 6], "tpy": [5, 6], "tpz": [5, 6], "te": [5, 6], "_p": 5, "to_numpi": 5, "transpos": 5, "formatspec": 5, "5f": 5, "savetxt": 5, "fmt": 5, "delimit": 5, "comment": 5, "ps_electron": 5, "test_new_export": 5, "__main__": 5, "0x7fb193ef7ca0": 5, "verifi": 5, "recycl": 5, "__": [5, 6], "wrote": 5, "newdataload": [5, 6], "_import_data": [5, 6], "loadtxt": [5, 6], "_input_data": [5, 6], "skiprow": [5, 6], "particle_properti": [5, 6, 8], "_particle_typ": [5, 6], "ones": [5, 6], "arang": [5, 6], "len": [5, 6], "replac": [5, 6], "convers": [5, 6], "consisten": [5, 6], "_check_energy_consist": [5, 6], "_check_input_data": [5, 6], "input": [5, 6], "is_fil": [5, 6], "rais": [5, 6], "filenotfounderror": [5, 6], "doe": [5, 6], "right": [5, 6], "extens": [5, 6], "except": [5, 6], "line": [5, 6], "correct": [5, 6], "open": [5, 6], "first_lin": [5, 6], "readlin": [5, 6], "good": 5, "One": 6, "easi": 6, "inhereit": 6, "_dataimportersbas": 6, "provid": 6, "whatev": 6, "suppli": 6, "etc": [6, 7], "pre": 6, "error": [6, 7], "__import_data": 6, "fundament": [6, 7], "popul": 6, "frame": 6, "attribut": 6, "succesfulli": 6, "guarante": 6, "am": 6, "go": 6, "develop": 6, "receiv": 6, "colleagu": 6, "encod": [6, 7], "text": 6, "alreadi": 6, "pretti": 6, "straight": 6, "includ": 6, "enter": 6, "below": [6, 7, 8], "complet": 6, "ok": 6, "new_data_loader_demo": 6, "down": 6, "__phase_space_config": 7, "split": 7, "categori": 7, "impli": 7, "core": 7, "free": 7, "allowed_column": 7, "order": 7, "associ": 7, "test_all_allowed_columns_can_be_fil": 7, "test_particlephasespac": 7, "descript": 7, "integ": 7, "statist": 7, "record": 7, "beta_x": 7, "beta_i": 7, "beta_z": 7, "beta_ab": 7, "lorentz": 7, "vx": 7, "vy": 7, "vz": 7, "chose": 7, "singl": 7, "framework": 7, "frequent": 7, "confus": 7, "simplest": 7, "safest": 7, "approach": 7, "seem": 7, "my": 7, "thought": 7, "prefer": 7, "clear": 7, "avoid": 7, "ambigu": 7, "hopefulli": 7, "trivial": 7, "help": 7, "properti": 7, "math": 7, "someth": 7, "p_x": 7, "directioncosine_x": 7, "rest_energi": 7, "511": [7, 8], "p_ev_c": 7, "p_si": 7, "p_mev_c": 7, "1e": 7, "rest_mass": 8, "602": 8, "22": 8, "proton": 8, "2212": 8, "938": 8, "272": 8, "neutron": 8, "2112": 8, "939": 8, "565": 8, "simpl": 8, "dictionari": 8, "some_new_particl": 8, "rest_mass_in_mev": 8, "charge_in_coulomb": 8, "alia": 8, "enabl": 8, "variou": 8, "part": 8, "pdg_code_new_particl": 8}, "objects": {"ParticlePhaseSpace": [[2, 0, 0, "-", "DataExporters"], [2, 0, 0, "-", "DataLoaders"], [2, 0, 0, "-", "_ParticlePhaseSpace"]], "ParticlePhaseSpace.DataExporters": [[2, 1, 1, "", "Topas_Exporter"]], "ParticlePhaseSpace.DataLoaders": [[2, 1, 1, "", "Load_PandasData"], [2, 1, 1, "", "Load_TibarayData"], [2, 1, 1, "", "Load_TopasData"]], "ParticlePhaseSpace._ParticlePhaseSpace": [[2, 1, 1, "", "PhaseSpace"]], "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace": [[2, 2, 1, "", "assess_density_versus_r"], [2, 2, 1, "", "calculate_twiss_parameters"], [2, 2, 1, "", "fill_beta_and_gamma"], [2, 2, 1, "", "fill_direction_cosines"], [2, 2, 1, "", "fill_kinetic_E"], [2, 2, 1, "", "fill_relativistic_mass"], [2, 2, 1, "", "fill_rest_mass"], [2, 2, 1, "", "fill_velocity"], [2, 2, 1, "", "filter_by_time"], [2, 2, 1, "", "get_downsampled_phase_space"], [2, 2, 1, "", "plot_beam_intensity"], [2, 2, 1, "", "plot_energy_histogram"], [2, 2, 1, "", "plot_n_particles_v_time"], [2, 2, 1, "", "plot_particle_positions"], [2, 2, 1, "", "plot_position_histogram"], [2, 2, 1, "", "plot_transverse_trace_space_intensity"], [2, 2, 1, "", "plot_transverse_trace_space_scatter"], [2, 2, 1, "", "print_energy_stats"], [2, 2, 1, "", "print_twiss_parameters"], [2, 2, 1, "", "project_particles"], [2, 2, 1, "", "reset_phase_space"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"express": 0, "bend": 0, "radiu": 0, "kinet": 0, "energi": [0, 7], "basic": 1, "exampl": [1, 3], "data": [1, 2, 5, 6], "import": 1, "analyt": 1, "seper": 1, "ad": [1, 8], "subtract": 1, "phase": [1, 2, 7], "space": [1, 2, 7], "object": 1, "twiss": 1, "paramet": 1, "manipul": 1, "export": [1, 2, 5], "code": 2, "document": 2, "loader": [2, 6], "particl": [2, 8], "particlephasespac": 4, "content": 4, "indic": 4, "tabl": 4, "write": [5, 6], "new": [5, 6, 8], "test": 5, "format": 7, "requir": 7, "column": 7, "allow": 7, "note": 7, "unit": 7, "read": 7, "momentum": 7, "If": 7, "direct": 7, "cosin": 7, "ar": 7, "specifi": 7, "beta": 7, "gamma": 7, "i": 7, "si": 7, "support": 8}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Express bending radius in kinetic energy": [[0, "express-bending-radius-in-kinetic-energy"]], "Basic Example": [[1, "Basic-Example"]], "Data import": [[1, "Data-import"]], "Basic analytics": [[1, "Basic-analytics"]], "Seperating, adding, and subtracting phase space objects": [[1, "Seperating,-adding,-and-subtracting-phase-space-objects"]], "Twiss parameters": [[1, "Twiss-parameters"]], "Manipulating data in the phase space": [[1, "Manipulating-data-in-the-phase-space"]], "Exporting the data": [[1, "Exporting-the-data"]], "Code Documentation": [[2, "code-documentation"]], "Data Loaders": [[2, "module-ParticlePhaseSpace.DataLoaders"]], "Particle Phase Space": [[2, "module-ParticlePhaseSpace._ParticlePhaseSpace"]], "Data Exporters": [[2, "module-ParticlePhaseSpace.DataExporters"]], "Examples": [[3, "examples"]], "Examples:": [[3, null]], "ParticlePhaseSpace": [[4, "particlephasespace"]], "Contents": [[4, "contents"]], "Contents:": [[4, null]], "Indices and tables": [[4, "indices-and-tables"]], "Writing a new data exporter": [[5, "Writing-a-new-data-exporter"]], "Testing the data export": [[5, "Testing-the-data-export"]], "Writing a new data loader": [[6, "Writing-a-new-data-loader"]], "Phase Space Format": [[7, "phase-space-format"]], "Required Columns": [[7, "required-columns"]], "Allowed Columns": [[7, "allowed-columns"]], "Notes on units": [[7, "notes-on-units"]], "Reading in momentum": [[7, "reading-in-momentum"]], "If energy/ direction cosines are specified:": [[7, "if-energy-direction-cosines-are-specified"]], "If beta/ gamma specified": [[7, "if-beta-gamma-specified"]], "If momentum is specified in SI units": [[7, "if-momentum-is-specified-in-si-units"]], "Supported particles": [[8, "supported-particles"]], "Adding new particles": [[8, "adding-new-particles"]]}, "indexentries": {"load_pandasdata (class in particlephasespace.dataloaders)": [[2, "ParticlePhaseSpace.DataLoaders.Load_PandasData"]], "load_tibaraydata (class in particlephasespace.dataloaders)": [[2, "ParticlePhaseSpace.DataLoaders.Load_TibarayData"]], "load_topasdata (class in particlephasespace.dataloaders)": [[2, "ParticlePhaseSpace.DataLoaders.Load_TopasData"]], "particlephasespace.dataexporters": [[2, "module-ParticlePhaseSpace.DataExporters"]], "particlephasespace.dataloaders": [[2, "module-ParticlePhaseSpace.DataLoaders"]], "particlephasespace._particlephasespace": [[2, "module-ParticlePhaseSpace._ParticlePhaseSpace"]], "phasespace (class in particlephasespace._particlephasespace)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace"]], "topas_exporter (class in particlephasespace.dataexporters)": [[2, "ParticlePhaseSpace.DataExporters.Topas_Exporter"]], "assess_density_versus_r() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.assess_density_versus_r"]], "calculate_twiss_parameters() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.calculate_twiss_parameters"]], "fill_beta_and_gamma() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.fill_beta_and_gamma"]], "fill_direction_cosines() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.fill_direction_cosines"]], "fill_kinetic_e() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.fill_kinetic_E"]], "fill_relativistic_mass() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.fill_relativistic_mass"]], "fill_rest_mass() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.fill_rest_mass"]], "fill_velocity() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.fill_velocity"]], "filter_by_time() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.filter_by_time"]], "get_downsampled_phase_space() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.get_downsampled_phase_space"]], "module": [[2, "module-ParticlePhaseSpace.DataExporters"], [2, "module-ParticlePhaseSpace.DataLoaders"], [2, "module-ParticlePhaseSpace._ParticlePhaseSpace"]], "plot_beam_intensity() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.plot_beam_intensity"]], "plot_energy_histogram() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.plot_energy_histogram"]], "plot_n_particles_v_time() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.plot_n_particles_v_time"]], "plot_particle_positions() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.plot_particle_positions"]], "plot_position_histogram() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.plot_position_histogram"]], "plot_transverse_trace_space_intensity() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.plot_transverse_trace_space_intensity"]], "plot_transverse_trace_space_scatter() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.plot_transverse_trace_space_scatter"]], "print_energy_stats() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.print_energy_stats"]], "print_twiss_parameters() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.print_twiss_parameters"]], "project_particles() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.project_particles"]], "reset_phase_space() (particlephasespace._particlephasespace.phasespace method)": [[2, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.reset_phase_space"]]}})