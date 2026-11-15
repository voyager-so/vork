// Generated from the Rust scene schema; do not edit.
export const resourceSlots = {
  "root": {
    "properties": {
      "animation": {
        "items": {
          "ref": "Layer"
        }
      },
      "buffers": {
        "values": {
          "ref": "Buffer"
        }
      },
      "composition": {
        "branches": [
          {
            "ref": "Composition"
          }
        ]
      },
      "geometry": {
        "values": {
          "ref": "Geometry"
        }
      },
      "images": {
        "values": {
          "ref": "Image"
        }
      },
      "materials": {
        "values": {
          "ref": "Material"
        }
      },
      "nodes": {
        "items": {
          "ref": "Node"
        }
      },
      "programs": {
        "values": {
          "ref": "Program"
        }
      }
    }
  },
  "defs": {
    "Audio": {
      "properties": {
        "file": {
          "file": true
        }
      }
    },
    "Buffer": {
      "branches": [
        {
          "properties": {
            "file": {
              "file": true
            }
          }
        },
        {
          "properties": {
            "file": {
              "file": true
            }
          }
        }
      ]
    },
    "Clip": {
      "properties": {
        "file": {
          "file": true
        }
      }
    },
    "CompositeEntry": {
      "branches": [
        {
          "ref": "NameOr"
        },
        {
          "ref": "CompositeNode"
        }
      ]
    },
    "CompositeNode": {
      "properties": {
        "material": {
          "ref": "NameOr"
        }
      }
    },
    "Composition": {
      "properties": {
        "compositor": {
          "items": {
            "ref": "CompositeEntry"
          }
        },
        "view": {
          "ref": "NameOr"
        }
      }
    },
    "Content": {
      "branches": [
        {
          "items": {
            "ref": "Run"
          }
        }
      ]
    },
    "Geometry": {
      "branches": [
        {
          "properties": {
            "attributes": {
              "values": {
                "ref": "NameOr3"
              }
            },
            "count": {
              "branches": [
                {
                  "ref": "NameOr3"
                }
              ]
            },
            "curve_vertex_counts": {
              "branches": [
                {
                  "ref": "NameOr3"
                }
              ]
            },
            "ids": {
              "branches": [
                {
                  "ref": "NameOr3"
                }
              ]
            },
            "indices": {
              "branches": [
                {
                  "ref": "NameOr3"
                }
              ]
            },
            "morphs": {
              "values": {
                "ref": "Morph"
              }
            },
            "status": {
              "branches": [
                {
                  "ref": "GeometryStatus"
                }
              ]
            },
            "widths": {
              "branches": [
                {
                  "ref": "NameOr3"
                }
              ]
            }
          }
        },
        {
          "properties": {
            "font": {
              "file": true
            }
          }
        }
      ]
    },
    "GeometryStatus": {
      "properties": {
        "buffer": {
          "ref": "NameOr3"
        }
      }
    },
    "GridBinding": {
      "properties": {
        "grid": {
          "ref": "NameOr3"
        }
      }
    },
    "Image": {
      "properties": {
        "composition": {
          "branches": [
            {
              "ref": "ImageComposition"
            }
          ]
        },
        "file": {
          "file": true
        }
      }
    },
    "ImageComposition": {
      "properties": {
        "compositor": {
          "items": {
            "ref": "CompositeEntry"
          }
        }
      }
    },
    "Input": {
      "branches": [
        {
          "ref": "TextureBinding"
        },
        {
          "ref": "GridBinding"
        }
      ]
    },
    "Instances": {
      "properties": {
        "attributes": {
          "values": {
            "ref": "NameOr3"
          }
        },
        "count": {
          "branches": [
            {
              "ref": "NameOr3"
            }
          ]
        }
      }
    },
    "Keys": {
      "branches": [
        {
          "properties": {
            "times": {
              "ref": "NameOr3"
            },
            "values": {
              "ref": "NameOr3"
            }
          }
        }
      ]
    },
    "Layer": {
      "branches": [
        {
          "properties": {
            "tracks": {
              "items": {
                "ref": "Track"
              }
            }
          }
        },
        {
          "properties": {
            "clip": {
              "branches": [
                {
                  "ref": "Clip"
                }
              ]
            }
          }
        }
      ]
    },
    "Material": {
      "properties": {
        "inputs": {
          "values": {
            "ref": "Input"
          }
        },
        "program": {
          "ref": "NameOr2"
        }
      }
    },
    "Morph": {
      "properties": {
        "NORMAL": {
          "branches": [
            {
              "ref": "NameOr3"
            }
          ]
        },
        "POSITION": {
          "ref": "NameOr3"
        }
      }
    },
    "NameOr": {
      "branches": [
        {
          "ref": "Material"
        }
      ]
    },
    "NameOr2": {
      "branches": [
        {
          "ref": "Program"
        }
      ]
    },
    "NameOr3": {
      "branches": [
        {
          "ref": "Buffer"
        }
      ]
    },
    "NameOr4": {
      "branches": [
        {
          "ref": "Image"
        }
      ]
    },
    "NameOr5": {
      "branches": [
        {
          "ref": "Geometry"
        }
      ]
    },
    "Node": {
      "branches": [
        {
          "properties": {
            "animation": {
              "items": {
                "ref": "NodeLayer"
              }
            },
            "audio": {
              "branches": [
                {
                  "ref": "Audio"
                }
              ]
            },
            "children": {
              "items": {
                "ref": "Node"
              }
            },
            "composition": {
              "branches": [
                {
                  "ref": "Composition"
                }
              ]
            }
          }
        },
        {
          "properties": {
            "animation": {
              "items": {
                "ref": "NodeLayer"
              }
            },
            "audio": {
              "branches": [
                {
                  "ref": "Audio"
                }
              ]
            },
            "children": {
              "items": {
                "ref": "Node"
              }
            },
            "composition": {
              "branches": [
                {
                  "ref": "Composition"
                }
              ]
            },
            "geometry": {
              "ref": "NameOr5"
            },
            "instances": {
              "branches": [
                {
                  "ref": "Instances"
                }
              ]
            },
            "materials": {
              "items": {
                "ref": "NameOr"
              }
            },
            "skin": {
              "branches": [
                {
                  "ref": "Skin"
                }
              ]
            }
          }
        },
        {
          "properties": {
            "animation": {
              "items": {
                "ref": "NodeLayer"
              }
            },
            "audio": {
              "branches": [
                {
                  "ref": "Audio"
                }
              ]
            },
            "children": {
              "items": {
                "ref": "Node"
              }
            },
            "composition": {
              "branches": [
                {
                  "ref": "Composition"
                }
              ]
            },
            "file": {
              "file": true
            },
            "materials": {
              "values": {
                "ref": "NameOr"
              }
            }
          }
        },
        {
          "properties": {
            "animation": {
              "items": {
                "ref": "NodeLayer"
              }
            },
            "audio": {
              "branches": [
                {
                  "ref": "Audio"
                }
              ]
            },
            "children": {
              "items": {
                "ref": "Node"
              }
            },
            "composition": {
              "branches": [
                {
                  "ref": "Composition"
                }
              ]
            },
            "material": {
              "branches": [
                {
                  "ref": "NameOr"
                }
              ]
            }
          }
        },
        {
          "properties": {
            "animation": {
              "items": {
                "ref": "NodeLayer"
              }
            },
            "audio": {
              "branches": [
                {
                  "ref": "Audio"
                }
              ]
            },
            "children": {
              "items": {
                "ref": "Node"
              }
            },
            "composition": {
              "branches": [
                {
                  "ref": "Composition"
                }
              ]
            },
            "image": {
              "branches": [
                {
                  "ref": "NameOr4"
                }
              ]
            },
            "material": {
              "branches": [
                {
                  "ref": "NameOr"
                }
              ]
            },
            "medium": {
              "branches": [
                {
                  "ref": "NameOr"
                }
              ]
            }
          }
        },
        {
          "properties": {
            "animation": {
              "items": {
                "ref": "NodeLayer"
              }
            },
            "audio": {
              "branches": [
                {
                  "ref": "Audio"
                }
              ]
            },
            "children": {
              "items": {
                "ref": "Node"
              }
            },
            "composition": {
              "branches": [
                {
                  "ref": "Composition"
                }
              ]
            },
            "content": {
              "ref": "Content"
            },
            "font": {
              "branches": [
                {
                  "file": true
                },
                {
                  "items": {
                    "file": true
                  }
                }
              ]
            }
          }
        },
        {
          "properties": {
            "animation": {
              "items": {
                "ref": "NodeLayer"
              }
            },
            "audio": {
              "branches": [
                {
                  "ref": "Audio"
                }
              ]
            },
            "children": {
              "items": {
                "ref": "Node"
              }
            },
            "composition": {
              "branches": [
                {
                  "ref": "Composition"
                }
              ]
            }
          }
        }
      ]
    },
    "NodeLayer": {
      "branches": [
        {
          "properties": {
            "tracks": {
              "items": {
                "ref": "Track"
              }
            }
          }
        },
        {
          "properties": {
            "clip": {
              "branches": [
                {
                  "ref": "Clip"
                }
              ]
            }
          }
        }
      ]
    },
    "OutputDestination": {
      "properties": {
        "file": {
          "file": true
        }
      }
    },
    "Program": {
      "branches": [
        {
          "properties": {
            "bindings": {
              "values": {
                "ref": "NameOr3"
              }
            },
            "file": {
              "file": true
            }
          }
        },
        {
          "properties": {
            "bindings": {
              "values": {
                "ref": "NameOr3"
              }
            },
            "source_origin": {
              "file": true
            }
          }
        }
      ]
    },
    "Run": {
      "properties": {
        "font": {
          "branches": [
            {
              "branches": [
                {
                  "file": true
                },
                {
                  "items": {
                    "file": true
                  }
                }
              ]
            }
          ]
        }
      }
    },
    "Skin": {
      "properties": {
        "inverse_bind_matrices": {
          "ref": "NameOr3"
        }
      }
    },
    "TextureBinding": {
      "properties": {
        "image": {
          "ref": "NameOr4"
        }
      }
    },
    "Track": {
      "properties": {
        "keys": {
          "ref": "Keys"
        }
      }
    }
  }
} as const;
