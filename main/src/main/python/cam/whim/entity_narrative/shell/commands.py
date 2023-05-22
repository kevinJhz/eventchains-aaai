from cmd import Cmd
from operator import itemgetter
import readline  # Just importing this makes the shell get nice shell features
import os
import traceback
from whim_common.data.coref import Entity, Mention
from cam.whim.entity_narrative.chains.document import Event


HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", "..", "..", "..", "..",
                            "local", "chain_model_shell_history")
# Limit stored history
readline.set_history_length(500)


class ModelShell(Cmd):
    """
    Terminal shell for querying narrative chain models.

    """
    prompt = "> "

    def __init__(self, model, model_name, *args, **kwargs):
        Cmd.__init__(self, *args, **kwargs)
        self.model_name = model_name
        self.model = model
        self.result_limit = 10

        readline.set_completer_delims(" ")

        # Environment for executing Python commands
        # May get updated as time goes on
        self.env = {
            'model': self.model
        }

    def do_EOF(self, line):
        """Exits the shell"""
        print "\nGoodbye"
        return True

    def do_info(self, line):
        """Outputs information about the model"""
        print "Model type: %s" % self.model.MODEL_TYPE_NAME
        print self.model.description

    @staticmethod
    def build_event(text, entities):
        """
        Builds an event using a mini-language for specifying events from the command line

        These are of the form:
            verb_lemma(arg0 [, arg1 [, prep, arg2 ]])
        Each arg is:
            En, where n is an entity number
            --, or just blank, meaning unspecified
            headword, just a single word head of a noun phrase
        """
        full_text = text
        text = text.strip()
        # Look for the first open bracket to get the verb lemma
        verb_lemma, __, text = text.partition("(")
        if not text:
            raise ModelShellError("events must be of the form: verb_lemma(arg0 [, arg1 [, prep, arg2 ]]). Got: %s" %
                                  full_text)
        text = text.strip(") ")
        # Look for commas separating the args
        arg_texts = text.split(",")
        if len(arg_texts) == 0:
            raise ModelShellError("event needs at least one argument (subject)")
        elif len(arg_texts) == 3:
            raise ModelShellError("event with indirect object also needs a preposition (third and fourth args)")
        elif len(arg_texts) > 4:
            raise ModelShellError("too many args for event: %s" % full_text)
        elif len(arg_texts) == 4:
            # Event with indirect object
            preposition = arg_texts[2]
            arg_texts = [arg_texts[0], arg_texts[1], arg_texts[3]]
        else:
            preposition = None

        # Process the arg texts to make event arguments
        args = []
        for arg_text in arg_texts:
            arg_text = arg_text.strip()
            if not arg_text or arg_text == "--":
                # Arg not specified
                args.append(None)
            elif arg_text.startswith("E"):
                # Try interpreting as an entity number
                try:
                    entity_num = int(arg_text[1:])
                except ValueError:
                    raise ModelShellError("entities must be given as En, where n is an entity number")
                if entity_num >= len(entities):
                    raise ModelShellError("not enough entities in list (tried to get entity %d)" % entity_num)
                args.append(entities[entity_num])
            else:
                # Just treat as a headword
                args.append(arg_text)
        # Fill in left out args with Nones
        args.extend([None] * (3 - len(args)))

        subj = args[0]
        obj = args[1]
        iobj = (preposition, args[2]) if args[2] is not None else None

        # Allow predicative adjectival events
        event_type = "predicative" if verb_lemma in ["be", "become"] else "normal"
        return Event(verb_lemma, verb_lemma, 0, type=event_type, subject=subj, object=obj, iobject=iobj)

    @staticmethod
    def build_entity(text):
        text = text.strip()
        # List of mentions, separated by semicolons
        mention_texts = [t.strip() for t in text.split(";")]
        # If no mention texts are given, put something in
        if not mention_texts:
            mention_texts = ["unnamed entity"]
        mentions = [Mention((0, len(t)), t, 0, 0, 1, 0, (0, len(t))) for t in mention_texts]
        # Currently don't allow specifying any other attributes, but this could easily be added
        return Entity(mentions)

    @staticmethod
    def parse_entity(line):
        if not line.startswith("E("):
            return None, line
        else:
            # Drop the open bracket
            line = line[2:]
            # Look for a close bracket
            entity_text, __, line = line.partition(")")
            # Get rid of the comma that (typically, but not required) follows
            line = line.lstrip(", ")
            # Build an entity out of the text in between
            entity = ModelShell.build_entity(entity_text)
            return entity, line

    @staticmethod
    def parse_event(line, entities=[], allow_arg_only=False):
        if allow_arg_only and ":" in line.partition(" ")[0]:
            # This looks like an argument-only spec
            arg_spec, __, line = line.partition(" ")
            arg_name, __, arg_text = arg_spec.partition(":")
            if arg_name not in ["arg0", "arg1", "arg2"]:
                raise ModelShellError("tried to interpret '%s' as an argument-only spec, but arg name should be "
                                      "arg0, arg1 or arg2. If you meant to specify a predicate, use the "
                                      "predicate-argument syntax" % arg_spec)
            return (arg_name, arg_text), line
        else:
            # Cut off everything after the event spec and keep it
            event_text, closer, line = line.partition(")")
            line = line.lstrip()
            return ModelShell.build_event(event_text, entities), line

    @staticmethod
    def parse_event_context(line, allow_arg_only=False):
        entities = []
        while line.startswith("E("):
            try:
                # Remove any entity from the beginning of the line and parse it
                entity, line = ModelShell.parse_entity(line)
            except ModelShellError, err:
                raise ModelShellError("error parsing entities: %s" % err)
            # Build an entity out of the text in between
            entities.append(entity)

        if not entities:
            # No entities were specified: make one that will be the chain entity
            entities.append(ModelShell.build_entity(""))

        line = line.strip()
        context_events = []
        while len(context_events) == 0 or line.startswith("~"):
            line = line.lstrip("~ ")
            try:
                event, line = ModelShell.parse_event(line, entities, allow_arg_only=allow_arg_only)
            except ModelShellError, err:
                raise ModelShellError("error parsing context events: %s" % err)
            context_events.append(event)

        return entities, context_events, line

    @staticmethod
    def parse_entities_and_event(line):
        return ModelShell.parse_entities_and_events(line, max_events=1)

    @staticmethod
    def parse_entities_and_events(line, max_events=None, allow_arg_only=False):
        entities = []
        while line.startswith("E("):
            # Remove any entity from the beginning of the line and parse it
            entity, line = ModelShell.parse_entity(line)
            # Build an entity out of the text in between
            entities.append(entity)

        if not entities:
            # No entities were specified: make one that will be the chain entity
            entities.append(ModelShell.build_entity(""))

        line = line.lstrip()
        events = []
        while line.strip():
            event, line = ModelShell.parse_event(line, entities, allow_arg_only=allow_arg_only)
            events.append(event)
            if max_events and len(events) == max_events:
                break

        return entities, events, line

    @staticmethod
    def parse_event_context_and_choices(line, allow_arg_only=False):
        entities, context_events, line = ModelShell.parse_event_context(line)
        choices = []
        line.lstrip(" ")
        # Next split up the list of alternative next events
        while len(choices) == 0 or line.startswith("|"):
            line = line.lstrip("| ")
            try:
                event, line = ModelShell.parse_event(line, entities, allow_arg_only=allow_arg_only)
            except ModelShellError, err:
                raise ModelShellError("error parsing choice event (from %s): %s" % (line, err))
            line.lstrip(" ")
            choices.append(event)
        return entities, context_events, choices

    def do_choose(self, line):
        """
        Syntax:
          event [E(...) [, E(...)]] [<context-event-0> [~ <context-event-1> ...]] [<completion-0> [~ <completion-1> ...]]

        Make predictions of the next event, given the events so far in the chain.

        First, (optionally) specify a list of entities.
        Next, a ~-separated list of events that build up the context.
        Finally, a |-separated list of alternative next events to choose from using the model.

        Use the event mini-language to specify events. Entity 0 is the chain entity, which defaults to as
        generic an entity as is possible. If you want to build the entity yourself, specify it before the
        event list. Other entities may also be built there, numbered in the order given.

        """
        entities, context_events, choices = ModelShell.parse_event_context_and_choices(line)

        # Score the choices as completions of the context, using the model
        scores = self.model.score_choices(entities[0], context_events, choices)

        sorted_scores = list(reversed(sorted(enumerate(scores), key=itemgetter(1))))
        print "Top choice: %d, %s" % (sorted_scores[0][0], choices[sorted_scores[0][0]])
        print "\nScores:"
        print "  %s" % "\n  ".join(["%.3g: %s" % (float(score), choices[choice_num])
                                    for (choice_num, score) in sorted_scores])

    def do_exec(self, line):
        """
        Execute a Python statement
        """
        exec line in self.env

    def do_x(self, line):
        """ Alias for `exec` """
        self.do_exec(line)

    def do_py(self, line):
        """
        Triggered by the vertical bar (and "py" command). Executes a python command and
        outputs the result.

        """
        result = eval(line, self.env)
        print result

    def preloop(self):
        # Load shell history
        if os.path.exists(HISTORY_FILE):
            readline.read_history_file(HISTORY_FILE)

    def postloop(self):
        # Save shell history
        readline.write_history_file(HISTORY_FILE)

    def emptyline(self):
        """ Don't repeat the last command (default): ignore empty lines """
        return

    @staticmethod
    def get_kwargs(arg):
        if arg is None:
            return {}, arg
        # Pull out key=value pairs from the front to use as kwargs
        w = 0
        words = arg.split()
        kwargs = {}
        for word in words:
            if "=" not in word:
                break
            key, __, value = word.partition("=")
            # Use this as a kwargs
            kwargs[key] = value
            # One word has been consumed: don't include it in arg
            w += 1
        arg = " ".join(words[w:])
        return kwargs, arg

    def onecmd(self, line):
        """
        Override to provide kwargs.

        """
        if line.startswith("|"):
            # Special command: translate into something we can use in a function name
            line = "py %s" % line[1:]

        cmd, arg, line = self.parseline(line)
        kwargs = {}
        if cmd not in ["exec", "x", "py"]:
            kwargs, arg = ModelShell.get_kwargs(arg)

        if not line:
            return self.emptyline()
        if cmd is None:
            return self.default(line)
        self.lastcmd = line
        if line == 'EOF' :
            self.lastcmd = ''
        if cmd == '':
            return self.default(line)
        else:
            try:
                func = getattr(self, 'do_' + cmd)
            except AttributeError:
                return self.default(line)
            return func(arg, **kwargs)

    def cmdloop(self, intro=None):
        if intro or self.intro:
            print intro or self.intro

        while True:
            try:
                Cmd.cmdloop(self, intro="")
            except ModelShellError, e:
                print e
            except KeyboardInterrupt:
                print
                self.postloop()
            except:
                # Print out the stack trace and return to the shell
                print "Error running command:"
                traceback.print_exc()
                self.postloop()
            else:
                self.postloop()
                break


class ModelShellError(Exception):
    pass
