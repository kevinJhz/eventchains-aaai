from __future__ import absolute_import
import argparse
import sys

from cam.whim.entity_narrative.models import MODEL_TYPES, get_model_class
from cam.whim.entity_narrative.models.base.model import NarrativeChainModel, cmd_line_model_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a shell to query models and such")
    parser.add_argument("model_type", help="Type of model to load", nargs="?")
    parser.add_argument("model_name", help="Name of model to load", nargs="?")
    parser.add_argument("--opts", help="Model options. Use 'help' to see a list")
    opts = parser.parse_args()

    model_type = opts.model_type
    if model_type is None:
        # Output a list of available model types
        print "Available model types:\n  %s" % "\n  ".join(sorted(MODEL_TYPES.keys()))
        sys.exit(0)
    elif model_type not in MODEL_TYPES:
        print "%s is not a valid model type" % model_type
        print "Valid model types: %s" % ", ".join(MODEL_TYPES.keys())
        sys.exit(1)
    elif opts.model_name is None:
        model_cls = get_model_class(model_type)
        # Output a list of the available models
        print "Available models of type %s:\n  %s" % (model_type,
                                                      "\n  ".join(sorted(model_cls.list_models())))
        sys.exit(0)

    # Process model options
    options = cmd_line_model_options(model_type, opts.opts)

    print "Loading %s model: %s" % (opts.model_type, opts.model_name)
    model = NarrativeChainModel.load_by_type(opts.model_type, opts.model_name, model_options=options)

    model.SHELL_TYPE(model, opts.model_name).cmdloop()
