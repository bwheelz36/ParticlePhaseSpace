Search.setIndex({"docnames": ["BendingRadius", "IAEA", "basic_example", "code_docs", "examples", "grid_merge", "index", "new_data_exporter", "new_data_loader", "phase_space_format", "resampling_via_gaussian_kde", "supported_particles", "transform", "units"], "filenames": ["BendingRadius.md", "IAEA.ipynb", "basic_example.ipynb", "code_docs.rst", "examples.rst", "grid_merge.ipynb", "index.rst", "new_data_exporter.ipynb", "new_data_loader.ipynb", "phase_space_format.md", "resampling_via_gaussian_kde.ipynb", "supported_particles.md", "transform.ipynb", "units.ipynb"], "titles": ["Express bending radius in kinetic energy", "Reading IAEA phase space", "Basic Example", "Code Documentation", "Examples", "Compress phase space via regrid/merge operations", "ParticlePhaseSpace", "Writing a new data exporter", "Writing a new data loader", "Phase Space Format", "Up/Down sampling phase space data using gaussian KDE", "Supported particles", "Transformation of phase space data", "Working with different units"], "terms": {"The": [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13], "particl": [0, 1, 2, 4, 6, 7, 8, 9, 12, 13], "magnet": 0, "field": [0, 1, 2, 3, 12], "i": [0, 1, 2, 3, 5, 7, 8, 10, 11, 12, 13], "given": [0, 2, 3, 13], "begin": 0, "equat": 0, "label": [0, 1, 13], "eqn": 0, "1": [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13], "tag": [0, 13], "r": [0, 1, 2, 3, 5, 8, 10, 12, 13], "frac": 0, "p": [0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "qb": 0, "gamma": [0, 1, 2, 5, 10, 11, 12], "m_0": 0, "v": [0, 1], "beta": [0, 2, 5, 10], "c": [0, 2, 3, 5, 7, 8, 9, 10, 11, 13], "end": [0, 1], "meanwhil": [0, 2], "relatavist": 0, "e_k": [0, 9], "mc": 0, "2": [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13], "m_0c": 0, "sqrt": [0, 1, 9], "our": [0, 1, 3, 7, 8, 10, 12], "goal": [0, 8], "term": [0, 9], "firstli": 0, "we": [0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "can": [0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "us": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13], "momentum": [0, 1, 2, 4, 6, 7, 8, 13], "relat": 0, "e": [0, 3, 7, 8, 9, 10, 13], "2c": 0, "4": [0, 1, 2, 5, 7, 8, 10, 12, 13], "ek": [0, 1, 2, 3, 7, 8, 9], "e0": 0, "e_0": [0, 9], "2e_k": 0, "now": [0, 1, 5, 8, 10, 12, 13], "assum": [0, 1, 10], "ev": [0, 9, 13], "have": [0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "just": [0, 1, 3, 7, 8, 10, 12, 13], "calcul": [0, 1, 3, 4, 5, 9, 12], "unit": [0, 1, 2, 3, 4, 6, 7, 8], "thi": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13], "directli": [0, 2, 10, 12], "would": [0, 1, 2, 7, 8, 9, 10, 12, 13], "convert": [0, 3, 7, 9, 13], "si": 0, "kg": [0, 9, 13], "m": [0, 3, 5, 9, 13], "": [0, 1, 2, 5, 7, 9, 10, 12, 13], "p_": 0, "q": [0, 3, 9], "substitut": 0, "bc": 0, "note": [0, 1, 2, 3, 4, 7, 8, 9, 10, 12, 13], "formula": 0, "electron": [0, 1, 2, 3, 5, 7, 8, 9, 11], "volt": 0, "rearrang": 0, "slightli": 0, "At": [0, 1, 3], "point": [0, 1, 3], "ar": [0, 1, 2, 3, 5, 7, 8, 10, 11, 12, 13], "still": [0, 10], "defin": [0, 2, 4, 9], "If": [0, 2, 3, 5, 7, 8, 13], "instead": [0, 1, 3, 13], "joul": 0, "bcq": 0, "which": [0, 1, 2, 3, 5, 8, 10, 12, 13], "same": [0, 2, 3, 5, 7, 12], "magdalena": 0, "answer": 0, "let": [0, 1, 2, 5, 7, 8, 10, 12, 13], "check": [0, 1, 2, 5, 7, 8, 10, 12], "all": [0, 1, 2, 3, 7, 8, 9, 10, 12, 13], "make": [0, 1, 2, 8, 9, 12], "sens": [0, 1, 2], "some": [0, 1, 2, 5, 7, 8, 9, 10, 12, 13], "basic": [0, 1, 3, 4, 6, 8], "import": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13], "numpi": [0, 1, 3, 5, 7, 8, 10], "np": [0, 1, 3, 5, 7, 8, 10], "from": [0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "scipi": 0, "constant": [0, 1, 3], "elementary_charg": 0, "b": [0, 10], "t": [0, 2, 3, 4, 5, 10, 12, 13], "10e6": 0, "ek_si": 0, "j": 0, "0": [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13], "511e6": 0, "e0_si": 0, "first": [0, 1, 2, 3, 5, 7, 8, 10, 12, 13], "version": 0, "r_brendan": 0, "r_magdalena": 0, "print": [0, 3, 10, 13], "f": [0, 1, 3, 7, 8, 10], "n": [0, 7, 8, 13], "nbend": 0, "nenergi": 0, "1e6": 0, "2f": 0, "mev": [0, 2, 3, 5, 7, 8, 9, 10, 11, 13], "ncharg": 0, "2e": [0, 1], "nin": 0, "ni": 0, "brendan": [0, 1, 3, 10], "3f": 0, "10": [0, 1, 2, 3, 5, 10, 13], "00": [0, 5, 10], "charg": [0, 9, 11], "60e": 0, "19": [0, 2, 11], "In": [0, 1, 2, 3, 5, 8, 9, 10, 12, 13], "018": 0, "onli": [0, 1, 2, 3, 7, 9, 10, 12, 13], "other": [0, 1, 2, 3, 4, 9, 13], "thing": [0, 4, 5], "worth": 0, "here": [0, 1, 2, 3, 5, 9, 12], "light": 0, "ratio": 0, "rest": [0, 2, 9], "becom": 0, "small": [0, 1, 3, 10], "high": 0, "second": [0, 1, 2, 10], "ha": [0, 1, 3, 8, 10, 12], "valu": [0, 1, 3, 5, 7, 8, 9, 10, 12, 13], "so": [0, 2, 3, 4, 5, 8, 10, 11, 13], "ignor": 0, "write": [0, 2, 3, 4, 6, 9], "much": [0, 1, 5, 10], "more": [0, 2, 10], "simpli": [0, 13], "approx": 0, "r_approxim": 0, "approxim": 0, "nwhich": 0, "wrong": 0, "100": [0, 5, 12], "ab": [0, 5, 10], "017": 0, "75": 0, "intern": [1, 3, 13], "atom": 1, "energi": [1, 2, 3, 4, 7, 8, 13], "agenc": 1, "maintain": 1, "databas": 1, "differ": [1, 2, 4, 6, 7, 8, 9, 10, 12], "medic": 1, "acceler": 1, "describ": [1, 2], "report": 1, "principl": [1, 10], "great": 1, "standardis": 1, "everyon": 1, "adher": 1, "realiti": 1, "loos": 1, "fratern": 1, "data": [1, 4, 5, 6, 9], "mean": [1, 2, 5, 8, 10, 13], "veri": [1, 5, 7, 10, 12, 13], "difficult": 1, "provid": [1, 8], "method": [1, 2, 3, 7, 8, 9, 10, 12, 13], "although": [1, 10], "dataload": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "load_iaea": [1, 3], "like": [1, 2, 3, 7, 8, 9, 10, 12, 13], "out": [1, 2, 3, 8, 10, 12], "box": 1, "arbitrari": [1, 3], "requir": [1, 3, 4, 6, 7, 8, 10, 12, 13], "fine": 1, "tune": 1, "therefor": [1, 12], "exampl": [1, 3, 5, 6, 7, 8, 9, 12, 13], "demonstr": [1, 2, 5, 8, 10], "process": [1, 5, 7], "you": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "go": [1, 5, 8, 10], "through": [1, 2, 3], "code": [1, 2, 5, 6, 8, 9, 11, 12, 13], "consist": [1, 2, 13], "two": [1, 2, 3, 8, 9, 10], "A": [1, 5, 10], "header": [1, 7, 8], "ascii": [1, 3, 8], "human": 1, "readabl": 1, "encod": [1, 8, 9], "contain": [1, 3, 8], "lot": [1, 2, 10], "inform": [1, 5], "essenti": 1, "how": [1, 2, 3, 5, 8, 9, 10, 12], "phasespac": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "typic": [1, 3], "extens": [1, 7, 8], "phsp": [1, 2, 3, 5, 7, 10, 12, 13], "iaeaphsp": [1, 3], "binari": [1, 3], "correctli": 1, "need": [1, 2, 5, 7, 8, 10, 13], "understand": 1, "exactli": [1, 12], "what": [1, 2, 5, 8, 9, 10, 13], "order": [1, 3, 9], "quantiti": [1, 3, 4, 5, 9, 10, 12, 13], "mani": [1, 2, 3, 5, 9], "byte": 1, "them": [1, 9, 10], "should": [1, 2, 3, 7, 8, 9, 10, 12], "avail": [1, 2, 4, 7, 8, 9, 12], "follow": [1, 2, 5, 7, 8, 9, 11, 12, 13], "tabl": 1, "howev": [1, 2, 5, 9, 10, 12, 13], "ani": [1, 2, 3, 5, 8, 9, 10, 12, 13], "also": [1, 2, 3, 7, 8, 9, 12, 13], "possibl": [1, 10, 12], "dervic": 1, "yet": 1, "clear": [1, 12], "me": 1, "whether": [1, 2], "thei": [1, 3, 8, 10, 12], "enough": [1, 10], "alwai": [1, 2, 3, 5, 9, 10], "allow": [1, 3, 6, 7, 10], "practic": [1, 5], "With": 1, "said": 1, "doe": [1, 7, 8, 13], "one": [1, 2, 3, 10, 12, 13], "actual": [1, 2, 7, 8, 10, 13], "about": [1, 5, 10, 13], "case": [1, 3, 5, 7, 9, 10, 12], "section": 1, "varian": [1, 3], "type": [1, 2, 3, 7, 8, 9, 13], "x": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "posit": [1, 2, 4, 9, 12], "y": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "compon": 1, "direct": [1, 3, 12], "vector": [1, 12], "record_const": 1, "26": 1, "7": [1, 2, 5, 10, 13], "z": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "0000": 1, "weight": [1, 2, 3, 5, 7, 8, 9, 10, 13], "piec": 1, "specifi": [1, 3, 7, 8], "sy": [1, 2, 5, 7, 8, 10, 12, 13], "path": [1, 2, 3, 5, 7, 8, 10, 12, 13], "insert": 1, "necessari": [1, 2, 5, 10, 12, 13], "instal": [1, 2, 5, 8, 10, 12, 13], "particlephasespac": [1, 2, 3, 5, 7, 8, 10, 11, 12, 13], "pathlib": [1, 2, 3, 5, 7, 8, 10, 12, 13], "get": [1, 2, 8, 10, 12], "file_nam": [1, 3], "home": [1, 3, 10], "dropbox": 1, "sydnei": 1, "uni": 1, "abstract": 1, "present": [1, 2, 3, 9], "etc": [1, 8, 9], "temp": 1, "truebeam_v2_6x_00": 1, "download": [1, 3], "varian_truebeam6mv_01": [1, 3], "data_schema": [1, 3], "dtype": [1, 3], "i1": [1, 3], "f4": [1, 3], "cosin": [1, 3], "float32": 1, "int8": [1, 3], "ps_data": [1, 2, 3, 5, 7, 10, 12, 13], "input_data": [1, 3], "n_record": [1, 3], "int": [1, 3], "1e5": [1, 3], "document": [1, 6, 9, 10, 11], "python": [1, 6, 10], "py": [1, 2, 3, 5, 7, 8, 9, 10, 11, 13], "440": 1, "userwarn": [1, 10], "neg": [1, 10], "even": [1, 5], "forc": [1, 12], "warn": [1, 10], "39": [1, 2, 10], "were": 1, "abl": 1, "figur": [1, 3], "loader": [1, 4, 6, 7], "few": [1, 5, 10], "control": 1, "row": [1, 3], "sinc": [1, 2, 5, 7, 8, 10, 12], "larg": [1, 3, 4, 5], "desir": [1, 3], "chunk": [1, 3, 10], "got": [1, 2], "As": [1, 2, 7, 10], "sai": [1, 7], "know": [1, 2, 10], "why": 1, "physic": 1, "ve": 1, "load": [1, 3, 7, 8, 10], "take": [1, 2, 10], "look": [1, 2, 7, 8, 10, 12, 13], "again": [1, 2, 10, 13], "becaus": [1, 2, 3, 5, 7, 8, 10, 12, 13], "manual": [1, 10, 12, 13], "delet": [1, 3], "finish": 1, "del": 1, "plot": [1, 2, 3, 5, 7, 8, 10, 12, 13], "particle_positions_hist_2d": [1, 2, 5, 10, 12], "log_scal": 1, "true": [1, 2, 3, 5, 8, 10, 12], "3": [1, 2, 5, 7, 8, 10, 12, 13], "energy_hist_1d": [1, 2, 5, 7, 10, 13], "right": [1, 7, 8], "suspect": 1, "spatial": 1, "coordin": [1, 12], "cm": [1, 13], "xy": 1, "intens": [1, 2], "suggest": 1, "beam": [1, 2, 3, 10, 12], "spread": [1, 2, 5, 10], "between": [1, 2, 10, 13], "mm": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "seem": 1, "To": [1, 7, 9], "chang": [1, 4, 10, 12], "particlephasespaceunit": [1, 3, 7, 13], "available_unit": [1, 3, 13], "cm_mev": [1, 13], "anoth": [1, 2, 12], "truebeam": 1, "instanc": [1, 2, 3, 8, 9, 12, 13], "nice": 1, "guess": [1, 10], "tell": 1, "u": [1, 3, 12], "record_cont": 1, "store": [1, 2, 8, 12], "w": 1, "extra": 1, "float": [1, 3], "long": [1, 10, 13], "base": [1, 3, 5, 10], "info": 1, "abov": [1, 2, 10, 11, 13], "scheme": 1, "someth": [1, 9], "i2": 1, "see": [1, 2, 3, 5, 9, 10, 12], "goe": 1, "8": [1, 2, 5, 10, 13], "except": [1, 7, 8], "traceback": 1, "most": [1, 5], "recent": 1, "call": [1, 2, 3, 8], "last": 1, "cell": 1, "line": [1, 2, 7, 8], "gt": [1, 2, 7], "397": 1, "__init__": [1, 13], "self": [1, 3, 7, 8, 13], "offset": [1, 3], "kwarg": [1, 3], "395": 1, "_n_record": 1, "396": [1, 2, 13], "_offset": 1, "super": 1, "53": 1, "_dataloadersbas": [1, 3, 7, 8], "particle_typ": [1, 3, 7, 8], "51": 1, "_input_data": [1, 7, 8], "52": 1, "_check_input_data": [1, 7, 8], "_import_data": [1, 7, 8], "54": 1, "_check_loaded_data": 1, "413": 1, "411": 1, "_header_to_dict": 1, "412": 1, "_check_required_info_pres": 1, "_check_record_length_versus_data_schema": 1, "414": 1, "415": 1, "_data_schema": 1, "554": 1, "552": 1, "record_length": 1, "_header_dict": 1, "kei": 1, "553": 1, "items": 1, "rais": [1, 7, 8], "schema": 1, "differeni": 1, "length": [1, 13], "indic": [1, 3], "555": 1, "nheader": 1, "34": 1, "556": 1, "nschema": 1, "25": [1, 2, 13], "interest": 1, "error": [1, 8], "There": [1, 13], "each": [1, 5, 9, 13], "entri": [1, 10], "Of": [1, 10], "cours": [1, 2, 10], "number": [1, 2, 5, 10, 13], "data_schem": 1, "specif": [1, 2, 5, 7], "match": [1, 7, 10, 13], "befor": [1, 2, 3, 5, 10], "try": [1, 5, 10], "anyth": 1, "els": [1, 7], "appear": [1, 5, 10], "longer": 1, "than": [1, 2, 9, 10], "previou": [1, 10], "been": [1, 2, 12], "known": 1, "integ": [1, 9], "could": [1, 5, 10], "explain": 1, "discrep": 1, "9": [1, 2, 5, 10, 13], "typeerror": 1, "11": [1, 2, 3, 5, 7, 10, 11, 13], "12": [1, 2, 5, 10, 13], "429": 1, "425": 1, "_column": [1, 8], "pd": [1, 3], "seri": [1, 3], "_constant": 1, "426": 1, "ones": [1, 7, 8, 9], "shape": [1, 2, 7, 8], "categori": [1, 9], "428": 1, "ok": [1, 5, 8, 10], "add": [1, 2, 5, 9, 11, 13], "particle_types_pdg": 1, "_iaea_types_to_pdg": 1, "430": 1, "_check_n_particles_in_head": 1, "431": 1, "477": 1, "varian_typ": 1, "475": 1, "uniqu": [1, 2, 5, 9, 10], "476": 1, "5": [1, 2, 5, 7, 8, 10, 12, 13], "unknown": 1, "478": 1, "pdg_type": 1, "22": [1, 2, 10, 11, 13], "479": 1, "74": 1, "do": [1, 2, 3, 5, 7, 10, 12, 13], "least": 1, "correct": [1, 7, 8, 13], "unidentifi": 1, "did": [1, 5], "time": [1, 2, 3, 5, 7, 8, 9, 10, 13], "13": [1, 2, 5, 10], "14": [1, 2], "458": 1, "456": 1, "locat": 1, "457": 1, "fail": [1, 10], "topa": [1, 2, 3, 7], "solut": 1, "increas": 1, "459": 1, "momentum_precision_factor": 1, "current": [1, 2, 10, 11, 13], "set": [1, 3, 4, 5, 7, 9, 10], "460": 1, "relative_differ": 1, "461": 1, "n_negative_loc": 1, "d": [1, 3, 5], "return": [1, 3], "invalid": [1, 2, 12], "pz": [1, 2, 3, 5, 7, 8, 9, 13], "zero": [1, 7, 8], "462": 1, "nwe": 1, "within": [1, 3], "463": 1, "_energy_consistency_check_cutoff": 1, "4f": 1, "_unit": [1, 13], "465": 1, "increaseth": 1, "00e": 1, "03and": 1, "71e": 1, "01": [1, 2, 5, 10, 13], "closer": [1, 10], "convers": [1, 7, 8, 13], "debug": 1, "wa": [1, 2, 3, 9, 12], "pretti": [1, 8, 10], "sure": 1, "6": [1, 2, 3, 5, 7, 8, 9, 10, 12, 13], "new": [1, 2, 3, 4, 6, 9, 10], "distribut": [1, 2, 10], "swap": 1, "around": [1, 5, 12], "column": [1, 3, 4, 6, 7, 8, 12, 13], "until": 1, "start": [1, 3, 12], "up": [1, 4, 6], "position_hist_1d": [1, 2, 5, 10, 12], "pass": [1, 2, 3, 7, 8, 9, 11, 12, 13], "expect": [1, 2, 7, 8, 12], "6mv": 1, "incorrect": 1, "develop": [1, 3, 5, 8], "autom": 1, "reader": 1, "frequent": 1, "wai": [1, 2, 3, 10, 12, 13], "For": [1, 2, 5, 8, 9, 10, 12, 13], "manag": 1, "after": [1, 4, 5, 12], "period": 1, "trial": 1, "It": [1, 3, 12], "realli": [1, 2, 10, 13], "step": [2, 4, 13], "analysi": [2, 3, 10], "handl": [2, 3, 8, 10], "an": [2, 3, 4, 7, 8, 9, 10, 11, 12], "class": [2, 3, 7, 8], "work": [2, 4, 5, 6, 7, 8, 9], "list": [2, 3, 7, 12, 13], "instruct": 2, "append": [2, 5, 7, 8, 10, 12, 13], "when": [2, 3, 4, 5, 10, 12, 13], "librari": [2, 3, 5, 6, 8, 10, 12, 13], "test_data_loc": [2, 5, 10, 12, 13], "test": [2, 3, 4, 5, 8, 9, 10, 11, 12, 13], "test_data": [2, 3, 5, 7, 8, 10, 12, 13], "coll_phasespace_xang_0": [2, 3, 5, 7, 10, 12, 13], "00_yang_0": [2, 3, 5, 7, 10, 12, 13], "00_angular_error_0": [2, 3, 5, 7, 10, 12, 13], "absolut": [2, 3, 5, 8, 10, 12, 13], "load_topasdata": [2, 3, 5, 7, 10, 12, 13], "onc": 2, "underli": [2, 3], "panda": [2, 3, 12], "frame": [2, 8], "shouldn": 2, "often": [2, 10], "want": [2, 3, 5, 7, 8, 9, 13], "interact": [2, 10, 12], "quick": 2, "head": [2, 13], "pdg_code": [2, 3, 7, 9, 11, 13], "id": [2, 3, 7, 8, 9, 13], "px": [2, 3, 5, 7, 8, 9, 10, 13], "971177": [2, 13], "817280": [2, 13], "008856": [2, 13], "010509": [2, 13], "184060": [2, 13], "916198": [2, 13], "839298": [2, 13], "003894": [2, 13], "008249": [2, 13], "560114": [2, 13], "258902": [2, 13], "318420": [2, 13], "193061": [2, 13], "377104": [2, 13], "219742": [2, 13], "15": [2, 13], "326971": [2, 13], "628899": [2, 13], "102889": [2, 13], "362594": [2, 13], "414843": [2, 13], "241948": [2, 13], "401026": [2, 13], "000192": [2, 13], "001470": [2, 13], "209232": [2, 13], "show": [2, 3, 9, 12], "minimum": 2, "quantati": 2, "variou": [2, 11], "fill": [2, 7, 8, 9], "kinet": [2, 9], "kinetic_": 2, "mass": [2, 9, 13], "p_ab": [2, 9], "184090": 2, "560141": 2, "291220": 2, "464187": 2, "209237": 2, "notic": 2, "while": [2, 5], "request": [2, 3, 12], "anywai": [2, 10], "alreadi": [2, 5, 7, 8, 10, 13], "recalcul": [2, 13], "view": 2, "command": 2, "get_method": [2, 12], "absolute_momentum": 2, "beta_and_gamma": 2, "direction_cosin": 2, "relativistic_mass": 2, "rest_mass": [2, 11], "veloc": [2, 9, 13], "access": 2, "insid": [2, 3, 9], "agnost": 2, "datafram": [2, 3], "without": [2, 5, 13], "refer": 2, "regardless": 2, "situat": [2, 9], "mai": [2, 3, 7, 8, 9, 10, 12], "wish": [2, 10, 12], "remov": [2, 3, 5, 12], "deriv": [2, 3, 9], "minimis": 2, "memori": [2, 3], "footpr": 2, "carri": [2, 12], "oper": [2, 3, 4, 6, 11, 12], "potenti": 2, "below": [2, 8, 9, 10, 11, 13], "reset_phase_spac": [2, 3, 10, 12, 13], "perform": [2, 3, 5], "print_energy_stat": [2, 3, 5, 10], "stat": [2, 3, 5, 10], "total": [2, 5, 9, 10], "311489": [2, 5], "speci": [2, 5, 10], "308280": [2, 5, 10], "91": [2, 5], "median": [2, 5, 10], "20": [2, 5], "iqr": [2, 5, 10], "03": [2, 5], "min": [2, 5, 10], "max": [2, 5, 10], "35": [2, 5, 10], "2853": [2, 5], "16": 2, "95": [2, 5], "02": [2, 5], "44": 2, "356": [2, 5], "positron": [2, 5, 11], "66": 2, "08": 2, "46": 2, "three": 2, "gener": [2, 3, 6, 7, 8, 9, 10], "particle_positions_scatter_2d": [2, 8], "beam_direct": [2, 3, 8], "color": 2, "visualis": 2, "plot_beam_intens": 2, "produc": 2, "imag": 2, "accumul": 2, "xlim": [2, 5, 10, 12], "ylim": [2, 5, 10, 12], "grid": [2, 5, 12], "illumin": 2, "scatter": [2, 5, 8], "score": 2, "exit": 2, "novel": 2, "rai": 2, "collim": 2, "squar": 2, "randomli": [2, 3, 10], "momentum_hist_1d": [2, 5, 10, 12], "n_particles_v_tim": 2, "transverse_trace_space_hist_2d": [2, 5, 10], "transverse_trace_space_scatter_2d": 2, "easili": [2, 9, 13], "electron_p": 2, "pdg": [2, 9], "multipl": 2, "gamma_p": 2, "positron_p": 2, "where": [2, 3, 5, 7, 8, 9, 10], "17": [2, 5], "no_gamma_p": 2, "togeth": [2, 5], "18": [2, 5, 10], "original_p": 2, "cannot": [2, 10], "ident": [2, 3, 5], "exist": [2, 3, 7, 8, 9, 10, 13], "updat": [2, 9, 11, 12, 13], "particle_id": 2, "courant": 2, "snyder": 2, "commonli": 2, "transvers": 2, "save": [2, 3], "print_twiss_paramet": [2, 3, 5, 10], "rm": [2, 3, 5], "epsilon": [2, 5, 10], "525901": [2, 5], "673361": [2, 5], "alpha": [2, 5, 10], "619585": [2, 5], "525727": [2, 5], "976938": [2, 5], "488084": [2, 5], "154160": [2, 5], "134526": [2, 5], "208669": 2, "357446": 2, "159332": 2, "009164": 2, "971102": [2, 5], "634599": [2, 5], "471535": [2, 5], "435510": [2, 5], "881200": [2, 5], "178092": [2, 5], "141023": [2, 5], "283977": 2, "888975": [2, 5], "672388": [2, 5], "591913": [2, 5], "095932": 2, "sometim": [2, 10], "trace": [2, 4], "versu": 2, "diverg": [2, 12], "red": 2, "ellips": [2, 5], "repres": [2, 9, 10, 13], "enclos": 2, "enclod": 2, "37": 2, "altern": 2, "sum": 2, "21": [2, 3], "plot_twiss_ellips": 2, "fals": [2, 3, 10, 12], "quit": [2, 4, 5, 10, 12], "fairli": [2, 3, 10], "evenli": [2, 12], "almost": 2, "highli": [2, 5, 13], "intesens": 2, "middl": 2, "og": 2, "log": 2, "scale": 2, "similarli": 2, "stage": 2, "dataexport": [2, 3, 7], "csv_export": [2, 3], "output_loc": [2, 3], "output_nam": [2, 3], "test_export": 2, "lt": [2, 7], "0x7fa5ecb4f700": 2, "page": [3, 6], "sphinx": 3, "automat": [3, 8, 13], "scan": 3, "sourc": 3, "dict": 3, "str": [3, 7], "iaea": [3, 4, 6], "sent": 3, "forum": 3, "format": [3, 4, 6, 7, 8], "variabl": 3, "pleas": 3, "paramet": [3, 4, 5, 10], "file": [3, 4, 7, 8, 9], "read": [3, 4, 6, 7, 8, 12], "By": [3, 13], "default": [3, 5, 13], "conjunct": 3, "load_pandasdata": 3, "none": [3, 7, 8, 9], "__unit_config__": [3, 13], "unitset": [3, 13], "object": [3, 4, 9, 10], "extern": 3, "dedic": [3, 12], "demo_data": 3, "load_tibaraydata": 3, "tibarai": 3, "rxy": 3, "bx": 3, "bz": 3, "g": [3, 9, 10, 13], "nmacro": 3, "rmacro": 3, "data_loc": [3, 7, 8], "tibaray_test": 3, "dat": [3, 7, 8], "both": [3, 8, 10], "behind": 3, "scene": 3, "reli": 3, "topas2numpi": 3, "load_p2sat_txt": 3, "adapt": 3, "p2sat": [3, 5], "txt": 3, "csv": 3, "um": [3, 13], "hard": [3, 10], "seper": [3, 4], "data_url": 3, "http": 3, "raw": 3, "githubusercont": 3, "com": 3, "lesnat": 3, "master": 3, "examplephasespac": 3, "p2sat_txt_test": 3, "urlretriev": 3, "p2_sat_uhi": [3, 13], "_particlephasespac": [3, 7, 8, 10], "data_load": 3, "hold": 3, "user": [3, 8, 10], "utilis": [3, 10], "common": 3, "accept": 3, "assess_density_versus_r": 3, "rval": 3, "verbos": 3, "bool": 3, "assess": 3, "radiu": 3, "linspac": 3, "screen": 3, "option": [3, 9, 10], "main": 3, "travel": [3, 12], "density_data": 3, "roi": 3, "val": 3, "proport": 3, "calculate_twiss_paramet": 3, "twiss": [3, 4, 5, 10], "filter_by_boolean_index": [3, 5, 10], "boolean_index": [3, 5, 10], "in_plac": [3, 5, 10, 12], "split": [3, 5, 9, 10, 12], "filter": [3, 10], "input": [3, 5, 7, 8], "boolean": 3, "index": [3, 6], "keep": 3, "discard": [3, 5, 10], "1d": 3, "arrai": 3, "structur": 3, "modifi": 3, "boolan_index": 3, "equal": 3, "filter_by_tim": 3, "t_start": 3, "t_finish": 3, "inclus": 3, "specfi": 3, "particleswith": 3, "new_inst": 3, "get_downsampled_phase_spac": [3, 10], "downsample_factor": 3, "randomlt": 3, "sampl": [3, 4, 6], "larger": [3, 10], "size": [3, 5], "origin": [3, 4, 5], "shuffl": 3, "being": [3, 5], "factor": [3, 9, 13], "downsampl": [3, 4], "merg": [3, 4, 6], "combin": 3, "regrid": [3, 4, 6], "algorithm": [3, 5], "leo": [3, 5], "esnault": [3, 5], "retain": 3, "group": 3, "new_p": [3, 5, 10], "summari": 3, "json": 3, "filenam": 3, "directori": 3, "resample_via_gaussian_kd": [3, 10], "n_new_particles_factor": [3, 10], "interpolate_weight": [3, 10], "fit": [3, 10], "gaussian": [3, 4, 6], "kernel": [3, 10], "densiti": [3, 10], "estim": [3, 10], "attempt": [3, 7, 9], "interpol": 3, "7d": 3, "experiment": [3, 10], "extrem": [3, 10], "caution": [3, 10], "len": [3, 7, 8, 10], "word": 3, "reduc": [3, 5, 12, 13], "_ps_data": 3, "whenev": [3, 10, 13], "footprint": 3, "set_unit": [3, 13], "new_unit": 3, "reset": [3, 4, 13], "sort": 3, "quantities_to_sort": 3, "accord": 3, "place": 3, "phasespaceinst": 3, "particular": [3, 12], "text": [3, 8], "topas_export": 3, "output": [3, 10], "featur": [3, 7, 8], "everi": [3, 13], "flag": 3, "histori": 3, "addit": [4, 8, 12], "phase": [4, 6, 7, 8], "space": [4, 6, 7, 8], "analyt": 4, "ad": [4, 5, 6], "subtract": 4, "export": [4, 6, 8], "don": [4, 10, 13], "well": [4, 10, 12], "next": 4, "transform": [4, 5, 6], "translat": 4, "rotat": 4, "project": 4, "less": 4, "support": [4, 6], "compress": [4, 6], "via": [4, 6], "conclus": 4, "down": [4, 6], "kde": [4, 6], "hoc": 4, "clean": 4, "compar": [4, 13], "simpl": [5, 11], "mechan": 5, "doesn": [5, 10], "tend": 5, "truli": 5, "close": [5, 10], "tutori": [5, 10, 12], "explot": 5, "fact": 5, "effect": 5, "nudg": 5, "bin": 5, "matplotlib": [5, 10], "pyplot": [5, 10], "plt": [5, 10], "vast": 5, "major": 5, "outsid": [5, 10], "rang": [5, 10], "come": [5, 13], "back": 5, "later": [5, 13], "keep_ind": [5, 10], "logical_and": [5, 10], "keep_ind2": 5, "keep_ind3": 5, "ps_discard": [5, 10], "account": [5, 10], "89": 5, "279094": 5, "took": 5, "henc": 5, "re": [5, 12], "n_bin": 5, "singl": 5, "157174": 5, "56": 5, "done": 5, "quantitiu": 5, "result": 5, "roughli": 5, "33": 5, "extract": 5, "previous": [5, 12], "obviou": 5, "question": 5, "50": 5, "82": 5, "07": 5, "93": 5, "80": 5, "154315": 5, "151106": 5, "09": 5, "83": 5, "92": 5, "49": 5, "208670": 5, "357445": 5, "159331": 5, "009165": 5, "283976": 5, "095931": 5, "690392": 5, "014616": 5, "443078": 5, "366593": 5, "144493": 5, "318489": 5, "194698": 5, "179535": 5, "211982": 5, "360174": 5, "157147": 5, "008641": 5, "963498": 5, "630658": 5, "471238": 5, "435652": 5, "885795": 5, "179988": 5, "139382": 5, "280596": 5, "882121": 5, "668230": 5, "591994": 5, "094013": 5, "includ": [5, 8], "stack": 5, "edg": 5, "strongli": 5, "affect": 5, "approach": [5, 10, 12], "lose": 5, "too": [5, 10], "extent": 5, "appropri": 5, "applic": [5, 10], "aim": 6, "serv": [6, 10], "purpos": [6, 8, 10, 12], "analys": 6, "manipul": [6, 12], "modul": [6, 12], "search": 6, "similar": 7, "those": [7, 10], "mm_mev": [7, 13], "creat": [7, 13], "_dataexportersbas": 7, "__particle_config__": [7, 8, 11], "particle_cfg": [7, 8], "newdataexport": 7, "def": [7, 8], "_define_required_column": 7, "_export_data": 7, "_set_expected_unit": 7, "job": [7, 8], "blank": 7, "must": [7, 8, 9, 13], "your": [7, 8, 10, 12, 13], "These": [7, 9, 10], "name": [7, 8, 9, 10, 11, 13], "set_expected_unit": 7, "receiv": [7, 8], "happen": [7, 10], "hand": 7, "_required_column": 7, "_expected_unit": 7, "_output_nam": 7, "suffix": [7, 8], "writefilepath": 7, "_output_loc": 7, "ty": [7, 8], "tz": [7, 8], "tpx": [7, 8], "tpy": [7, 8], "tpz": [7, 8], "te": [7, 8], "_p": 7, "to_numpi": 7, "transpos": 7, "formatspec": 7, "5f": 7, "savetxt": 7, "fmt": 7, "delimit": 7, "comment": 7, "ps_electron": 7, "test_new_export": 7, "__main__": 7, "0x7f5ad9d3a9e0": 7, "verifi": 7, "recycl": 7, "__": [7, 8], "wrote": 7, "newdataload": [7, 8], "loadtxt": [7, 8], "skiprow": [7, 8], "particle_properti": [7, 8, 11], "_particle_typ": [7, 8], "arang": [7, 8], "replac": [7, 8], "doubl": [7, 8], "consisten": [7, 8], "_check_energy_consist": [7, 8], "is_fil": [7, 8], "filenotfounderror": [7, 8], "open": [7, 8], "first_lin": [7, 8], "readlin": [7, 8], "good": [7, 10], "One": 8, "easi": [8, 13], "simul": [8, 10, 12], "inhereit": 8, "_dataimportersbas": 8, "whatev": 8, "suppli": 8, "pre": [8, 13], "__import_data": 8, "fundament": [8, 9, 10], "popul": 8, "attribut": 8, "succesfulli": 8, "guarante": [8, 10], "am": [8, 10], "colleagu": 8, "straight": 8, "forward": [8, 12], "enter": 8, "complet": [8, 10, 13], "new_data_loader_demo": 8, "weight_position_plot": 8, "slow": 8, "__phase_space_config": 9, "impli": [9, 10], "core": 9, "valid": [9, 10, 12], "fill_kinetic_": 9, "free": 9, "allowed_column": 9, "associ": 9, "test_all_allowed_columns_can_be_fil": 9, "test_particlephasespac": 9, "descript": 9, "relativist": 9, "statist": 9, "record": 9, "fill_rest_mass": 9, "fill_absolute_momentum": 9, "fill_relativistic_mass": 9, "beta_x": 9, "beta_i": 9, "beta_z": 9, "beta_ab": 9, "fill_beta_and_gamma": 9, "lorentz": 9, "vx": 9, "vy": 9, "vz": 9, "fill_veloc": 9, "fill_direction_cosin": 9, "sever": 9, "rather": 9, "live": 9, "notebook": 9, "hopefulli": 9, "trivial": 9, "help": 9, "properti": 9, "math": 9, "p_x": 9, "directioncosine_x": 9, "rest_energi": 9, "511": [9, 11], "form": [9, 13], "somewhat": [9, 12], "unusu": 9, "per": 9, "p_ev_c": 9, "p_si": 9, "p_mev_c": 9, "1e": [9, 13], "othertim": 10, "qualiti": 10, "mont": 10, "carlo": 10, "independ": 10, "primari": 10, "expens": 10, "task": 10, "solv": 10, "issu": 10, "further": 10, "heurist": 10, "undertak": 10, "care": [10, 13], "comparison": 10, "intend": 10, "dens": 10, "argu": 10, "my": 10, "properli": 10, "probabl": 10, "preserv": [10, 13], "sophist": 10, "function": 10, "challen": 10, "problem": 10, "seven": 10, "dimens": 10, "correl": 10, "popular": 10, "choic": 10, "sensit": 10, "tail": 10, "prone": 10, "idea": 10, "nearli": [10, 12], "li": 10, "filter_input_data": 10, "spatial_cutoff": 10, "86": 10, "model": 10, "own": [10, 12, 13], "littl": 10, "265149": 10, "132574": 10, "1488": 10, "old": 10, "ensur": [10, 13], "implement": 10, "improv": 10, "turn": 10, "off": 10, "aren": [10, 12], "Such": 10, "fashion": 10, "leav": 10, "individu": 10, "appli": 10, "big": 10, "partilc": 10, "histogram": 10, "haven": 10, "shown": 10, "reflect_pz": 10, "recomend": 10, "bit": 10, "disast": 10, "outlier": 10, "caus": 10, "higher": 10, "maximum": 10, "remove_high_divergence_particl": 10, "div_x": 10, "div_i": 10, "max_div_x": 10, "max_div_i": 10, "rem_ind": 10, "logical_or": 10, "logical_not": 10, "089075": 10, "089526": 10, "568640": 10, "563796": 10, "168591": 10, "133908": 10, "214531": 10, "214849": 10, "200457": 10, "180009": 10, "268318": 10, "286268": 10, "013035": 10, "386178": 10, "355786": 10, "319519": 10, "emitt": 10, "perfectli": 10, "terribl": 10, "anymor": 10, "either": [10, 11], "mang": 10, "23": 10, "131579": 10, "42": 10, "78": 10, "reason": 10, "excercis": 10, "badli": 10, "upsampl": 10, "mention": 10, "consid": 10, "accuraci": 10, "specfic": 10, "602": 11, "proton": 11, "2212": 11, "938": 11, "272": 11, "neutron": 11, "2112": 11, "939": 11, "565": 11, "dictionari": 11, "some_new_particl": 11, "rest_mass_in_mev": 11, "charge_in_coulomb": 11, "definit": [11, 13], "alia": 11, "enabl": 11, "part": 11, "pdg_code_new_particl": 11, "research": 12, "axi": 12, "system": 12, "occur": 12, "ps_translat": 12, "distanc": 12, "ps_rotat": 12, "rotation_axi": 12, "angl": 12, "45": 12, "succes": 12, "degre": 12, "keyword": 12, "rotate_momentum_vector": 12, "entir": 12, "accordingli": 12, "under": 12, "assumpt": 12, "experi": 12, "ps_project": 12, "significantli": 12, "wherea": 12, "exhibit": 12, "substanti": 12, "broaden": 12, "limit": 12, "ax": 12, "complex": 12, "cover": 12, "final": 12, "risk": 12, "edit": [12, 13], "underl": 12, "varieti": 13, "convent": 13, "program": 13, "m_ev": 13, "um_kev": 13, "kev": 13, "pick": 13, "my_unit": 13, "equivalal": 13, "unti": 13, "different_unit": 13, "get_unit": 13, "971": 13, "176758": 13, "817": 13, "280090": 13, "396000": 13, "855738": 13, "508804": 13, "3184": 13, "060481": 13, "916": 13, "197754": 13, "839": 13, "297668": 13, "894435": 13, "249210": 13, "1560": 13, "114059": 13, "5258": 13, "901367": 13, "11318": 13, "419922": 13, "193": 13, "061367": 13, "377": 13, "103855": 13, "1219": 13, "741850": 13, "15326": 13, "970703": 13, "25628": 13, "898438": 13, "102": 13, "888760": 13, "362": 13, "594239": 13, "1414": 13, "843426": 13, "241": 13, "947769": 13, "401": 13, "025604": 13, "191501": 13, "469658": 13, "209": 13, "231939": 13, "graph": 13, "automatical": 13, "advis": 13, "copi": 13, "unit_set": 13, "subject": 13, "dimensionless": 13, "prefer": 13, "cm_kev": 13, "length_unit": 13, "energy_unit": 13, "1e3": 13, "momentum_unit": 13, "time_unit": 13, "mass_unit": 13, "velocity_unit": 13, "kmph": 13, "mainli": 13, "compris": 13, "explanatori": 13, "THe": 13, "converion": 13, "value_in_mm": 13, "value_in_cm": 13, "cm_convers": 13, "Be": 13, "defint": 13, "substantiali": 13}, "objects": {"ParticlePhaseSpace": [[3, 0, 0, "-", "DataExporters"], [3, 0, 0, "-", "DataLoaders"], [3, 0, 0, "-", "_ParticlePhaseSpace"]], "ParticlePhaseSpace.DataExporters": [[3, 1, 1, "", "CSV_Exporter"], [3, 1, 1, "", "Topas_Exporter"]], "ParticlePhaseSpace.DataLoaders": [[3, 1, 1, "", "Load_IAEA"], [3, 1, 1, "", "Load_PandasData"], [3, 1, 1, "", "Load_TibarayData"], [3, 1, 1, "", "Load_TopasData"], [3, 1, 1, "", "Load_p2sat_txt"]], "ParticlePhaseSpace._ParticlePhaseSpace": [[3, 1, 1, "", "PhaseSpace"]], "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace": [[3, 2, 1, "", "assess_density_versus_r"], [3, 2, 1, "", "calculate_twiss_parameters"], [3, 2, 1, "", "filter_by_boolean_index"], [3, 2, 1, "", "filter_by_time"], [3, 2, 1, "", "get_downsampled_phase_space"], [3, 2, 1, "", "merge"], [3, 2, 1, "", "print_energy_stats"], [3, 2, 1, "", "print_twiss_parameters"], [3, 2, 1, "", "resample_via_gaussian_kde"], [3, 2, 1, "", "reset_phase_space"], [3, 2, 1, "", "set_units"], [3, 2, 1, "", "sort"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"]}, "titleterms": {"express": 0, "bend": 0, "radiu": 0, "kinet": 0, "energi": [0, 5, 9, 10], "read": [1, 9, 13], "iaea": 1, "phase": [1, 2, 3, 5, 9, 10, 12], "space": [1, 2, 3, 5, 9, 10, 12], "The": 1, "format": [1, 9], "an": 1, "file": 1, "when": 1, "thing": 1, "don": 1, "t": 1, "work": [1, 13], "quit": 1, "so": 1, "well": 1, "next": 1, "step": [1, 10], "basic": 2, "exampl": [2, 4, 10], "data": [2, 3, 7, 8, 10, 12, 13], "import": 2, "calcul": 2, "addit": 2, "quantiti": 2, "reset": 2, "requir": [2, 9], "column": [2, 9], "analyt": 2, "seper": 2, "ad": [2, 10, 11], "subtract": 2, "object": 2, "twiss": 2, "paramet": 2, "export": [2, 3, 7], "code": 3, "document": 3, "loader": [3, 8], "particl": [3, 5, 10, 11], "compress": 5, "via": 5, "regrid": [5, 12], "merg": 5, "oper": 5, "posit": [5, 10], "momentum": [5, 9, 10, 12], "trace": 5, "note": 5, "conclus": 5, "particlephasespac": 6, "content": 6, "indic": 6, "tabl": 6, "write": [7, 8], "new": [7, 8, 11, 13], "test": 7, "allow": 9, "unit": [9, 13], "If": 9, "direct": 9, "cosin": 9, "ar": 9, "specifi": 9, "beta": 9, "gamma": 9, "i": 9, "si": 9, "up": 10, "down": 10, "sampl": 10, "us": 10, "gaussian": 10, "kde": 10, "downsampl": 10, "larg": 10, "clean": 10, "input": 10, "hoc": 10, "reflect": 10, "pz": 10, "remov": 10, "high": 10, "diverg": 10, "compar": 10, "origin": 10, "support": [11, 12], "transform": 12, "translat": 12, "rotat": 12, "project": 12, "other": 12, "less": 12, "differ": 13, "avail": 13, "set": 13, "chang": 13, "after": 13, "defin": 13}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Express bending radius in kinetic energy": [[0, "express-bending-radius-in-kinetic-energy"]], "Reading IAEA phase space": [[1, "Reading-IAEA-phase-space"]], "The IAEA format": [[1, "The-IAEA-format"]], "Reading an IAEA phase space file": [[1, "Reading-an-IAEA-phase-space-file"]], "When things don\u2019t work quite so well\u2026": [[1, "When-things-don't-work-quite-so-well..."]], "Next steps": [[1, "Next-steps"]], "Basic Example": [[2, "Basic-Example"]], "Data import": [[2, "Data-import"]], "Calculation of additional quantities": [[2, "Calculation-of-additional-quantities"]], "Reset the phase space to required columns": [[2, "Reset-the-phase-space-to-required-columns"]], "Basic analytics": [[2, "Basic-analytics"]], "Seperating, adding, and subtracting phase space objects": [[2, "Seperating,-adding,-and-subtracting-phase-space-objects"]], "Twiss parameters": [[2, "Twiss-parameters"]], "Exporting the data": [[2, "Exporting-the-data"]], "Code Documentation": [[3, "code-documentation"]], "Data Loaders": [[3, "module-ParticlePhaseSpace.DataLoaders"]], "Particle Phase Space": [[3, "module-ParticlePhaseSpace._ParticlePhaseSpace"]], "Data Exporters": [[3, "module-ParticlePhaseSpace.DataExporters"]], "Examples": [[4, "examples"]], "Examples:": [[4, null]], "Compress phase space via regrid/merge operations": [[5, "Compress-phase-space-via-regrid/merge-operations"]], "Particle Positions": [[5, "Particle-Positions"], [10, "Particle-Positions"]], "Energy": [[5, "Energy"], [10, "Energy"]], "Momentum": [[5, "Momentum"], [10, "Momentum"]], "Trace Space": [[5, "Trace-Space"]], "Notes/ Conclusions": [[5, "Notes/-Conclusions"]], "ParticlePhaseSpace": [[6, "particlephasespace"]], "Contents:": [[6, null]], "Indices and tables": [[6, "indices-and-tables"]], "Writing a new data exporter": [[7, "Writing-a-new-data-exporter"]], "Testing the data export": [[7, "Testing-the-data-export"]], "Writing a new data loader": [[8, "Writing-a-new-data-loader"]], "Phase Space Format": [[9, "phase-space-format"]], "Required Columns": [[9, "required-columns"]], "Allowed Columns": [[9, "allowed-columns"]], "Units": [[9, "units"]], "Reading in momentum": [[9, "reading-in-momentum"]], "If energy/ direction cosines are specified:": [[9, "if-energy-direction-cosines-are-specified"]], "If beta/ gamma specified": [[9, "if-beta-gamma-specified"]], "If momentum is specified in SI units": [[9, "if-momentum-is-specified-in-si-units"]], "Up/Down sampling phase space data using gaussian KDE": [[10, "Up/Down-sampling-phase-space-data-using-gaussian-KDE"]], "Example: Downsampling a large phase space": [[10, "Example:-Downsampling-a-large-phase-space"]], "Clean input data": [[10, "Clean-input-data"]], "Ad-hoc data cleaning steps": [[10, "Ad-hoc-data-cleaning-steps"]], "Reflect Pz": [[10, "Reflect-Pz"]], "Remove high divergence particles": [[10, "Remove-high-divergence-particles"]], "Compare original to downsampled:": [[10, "Compare-original-to-downsampled:"]], "Supported particles": [[11, "supported-particles"]], "Adding new particles": [[11, "adding-new-particles"]], "Transformation of phase space data": [[12, "Transformation-of-phase-space-data"]], "Translations": [[12, "Translations"]], "Rotations": [[12, "Rotations"]], "Rotation of momentum": [[12, "Rotation-of-momentum"]], "Projection": [[12, "Projection"]], "Regridding": [[12, "Regridding"]], "Other (less supported!) transformations": [[12, "Other-(less-supported!)-transformations"]], "Working with different units": [[13, "Working-with-different-units"]], "Available units": [[13, "Available-units"]], "Setting units at data read in": [[13, "Setting-units-at-data-read-in"]], "Changing units after read in": [[13, "Changing-units-after-read-in"]], "Defining new unit sets": [[13, "Defining-new-unit-sets"]]}, "indexentries": {"csv_exporter (class in particlephasespace.dataexporters)": [[3, "ParticlePhaseSpace.DataExporters.CSV_Exporter"]], "load_iaea (class in particlephasespace.dataloaders)": [[3, "ParticlePhaseSpace.DataLoaders.Load_IAEA"]], "load_pandasdata (class in particlephasespace.dataloaders)": [[3, "ParticlePhaseSpace.DataLoaders.Load_PandasData"]], "load_tibaraydata (class in particlephasespace.dataloaders)": [[3, "ParticlePhaseSpace.DataLoaders.Load_TibarayData"]], "load_topasdata (class in particlephasespace.dataloaders)": [[3, "ParticlePhaseSpace.DataLoaders.Load_TopasData"]], "load_p2sat_txt (class in particlephasespace.dataloaders)": [[3, "ParticlePhaseSpace.DataLoaders.Load_p2sat_txt"]], "particlephasespace.dataexporters": [[3, "module-ParticlePhaseSpace.DataExporters"]], "particlephasespace.dataloaders": [[3, "module-ParticlePhaseSpace.DataLoaders"]], "particlephasespace._particlephasespace": [[3, "module-ParticlePhaseSpace._ParticlePhaseSpace"]], "phasespace (class in particlephasespace._particlephasespace)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace"]], "topas_exporter (class in particlephasespace.dataexporters)": [[3, "ParticlePhaseSpace.DataExporters.Topas_Exporter"]], "assess_density_versus_r() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.assess_density_versus_r"]], "calculate_twiss_parameters() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.calculate_twiss_parameters"]], "filter_by_boolean_index() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.filter_by_boolean_index"]], "filter_by_time() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.filter_by_time"]], "get_downsampled_phase_space() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.get_downsampled_phase_space"]], "merge() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.merge"]], "module": [[3, "module-ParticlePhaseSpace.DataExporters"], [3, "module-ParticlePhaseSpace.DataLoaders"], [3, "module-ParticlePhaseSpace._ParticlePhaseSpace"]], "print_energy_stats() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.print_energy_stats"]], "print_twiss_parameters() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.print_twiss_parameters"]], "resample_via_gaussian_kde() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.resample_via_gaussian_kde"]], "reset_phase_space() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.reset_phase_space"]], "set_units() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.set_units"]], "sort() (particlephasespace._particlephasespace.phasespace method)": [[3, "ParticlePhaseSpace._ParticlePhaseSpace.PhaseSpace.sort"]]}})