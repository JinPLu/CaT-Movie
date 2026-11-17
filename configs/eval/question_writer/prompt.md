You prepare checklist questions for judging a generated multi-shot video against the user request below.
A structured annotation of the same story is attached: its ids match the request and help you keep people and objects apart. The request alone decides what is required.

1. Names. For each registered id, give the name the request uses for it. If the request defines ids such as "Subject 2" or "Location 1", use the id itself as the name. Otherwise use the short phrase the request uses ("the birthday woman").

2. Shots. For each shot of the request, list the ids of the people and objects the request says are in that shot. Count one as in the shot only when the request mentions it there (by id, name, description or pronoun).

3. Actions. List every action the request describes that acts on an object or on another person, or moves the whole body (standing up, sitting down, walking away, falling). Steps that serve one goal are one action ("raises the glass, takes a sip and lowers it" is drinking). Leave out facial expressions, gaze, speaking, listening, camera and background. Only list actions of registered people or objects. Write each action as a short yes/no question about that one action alone, using the names from step 1 (good: "Does Subject 1 drink the red wine?"; not: "Does Subject 1 clap after blowing out the candles?"), and list the ids taking part.

   If the action changes something that stays changed afterwards (the candles are out, the cap is off, the box is now in her hands), also write one yes/no question about that new state, say whose state it is, and whether the change is to how someone or something looks ("appearance") or to who holds what or where it is ("holding_or_place"). Otherwise give null.
