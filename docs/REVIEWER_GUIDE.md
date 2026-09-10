# Reviewer Guide: Validation Part Two (segmentation fixes)

This guide builds on the main README. Read the README first for how to install and
launch the app. This is the extra step for the second round of review.

## Why we are doing this

In the first round we checked the functional attributes of each instance. While
doing that we noticed the segmentation and the class labels coming from the pipeline
are sometimes wrong. This round is about fixing those. So on top of confirming the
attributes, you now also correct the instance itself: give it the right class, split
it if it should be more than one object, or merge pieces that belong together.

## Where this data comes from

These files are built on top of validation part one. The attributes you see on each
instance are already the part-one corrected values, not the raw pipeline guesses. So
you are not starting from scratch. You are continuing from what was already validated,
and your job here is mainly the segmentation and class fixes on top of that.

## Before you start

> **IMPORTANT — delete your old bundle first.** If you opened one of these buildings
> in the app before, delete its old bundle folder (`bundles/<building_name>`) before you
> start, or the app will reuse the old version and you will NOT see the part-one
> attributes, flags, or notes. Once deleted, the app rebuilds it automatically from the
> new cloud the first time you open it.

Open each building one at a time. Look at it next to the imagery for that building
(the photos are on the AI server, in the usual place). If the photos are not clear
enough, open the building in Google Maps or Apple Maps and use their street/aerial
views to see what the object actually is.

## How to work: go class by class

It is much easier to filter the list to one class at a time (use the class dropdown)
and go through them, rather than jumping around. For each instance ask: is this the
right class, is it one object or several, and is it a clean object or a broken piece.

Use "Colour by" and pay attention to the colours and the flat planes. That is how you
spot where one object was split into pieces or where two objects got merged into one.

**Do not guess from the point cloud alone.** The cloud can be sparse or noisy, and a
shape you think is a window might not be one. Before you change anything, confirm what
the object really is against the building imagery, and if that is not clear enough use
Google Maps or Apple Maps. Only make the fix once you are sure. Eyeballing it and
assuming is how mistakes get in.

## Finding instances in the list

- **Search box:** type an id or a class name. You can look up several at once by
  separating them with commas, for example `12, 45, 80`, or mix in a class like
  `12, blinds`.
- **Filter checkboxes** above the list:
  - **unreviewed** hides the ones you already confirmed.
  - **flagged** shows only instances that carry a part-one flag.
  - **changed** shows only instances whose attributes differ from what came in.
  - **recent** shows the last instances you edited, most recent first, so you can
    jump straight back to what you were just working on.
- **Confirm ✓ & Next** marks the selected instance reviewed and moves you forward to
  the next one, so you can work through a building in order without losing your place.

## Part-one flags and notes

Some instances already carry a flag from part one, shown with a small flag mark in the
list, and some carry a note. These come from the first round, where reviewers marked
instances they thought had the wrong class or bad segmentation and wrote down why. Use
the "flagged" filter to jump straight to them.

Read the notes. A note like "this should be a window" tells you exactly what to fix, and
the flags point you at the problems part one already spotted. They are the best starting
hints for where the segmentation issues are.

But they are not the full list. Part one was focused on the attributes, so not every
segmentation or class error was caught, and some buildings were not flagged at all. Do
not only fix the flagged ones. Go through every instance, class by class, and check it
yourself. The flags and notes point you at the known problems. You are here to find the
rest too.

## The three things to fix

1. **Wrong class.** If an instance is labelled as the wrong thing, change its class
   with the Class dropdown. Example: a blind that is really a window.

2. **Over or under segmentation.** If one instance actually covers several separate
   objects, use Lasso split to cut it into parts and give each part its class.

3. **Fragmentation.** If one real object got broken into several instances, select
   them and Merge. Match the colours and watch the planes to be sure the pieces
   really belong to the same object.

## Leftover (unsegmented) points

A few points in each building were never assigned to any object. They show up as one
entry in the list called **unsegmented** (id -1). If some of those points really belong
to an object, select `unsegmented`, lasso that cluster, and either split it into its own
instance or merge it into the object it belongs to.

## Where the tools are and how they work

- **Class dropdown** (right panel, next to the Flag box). Pick the correct class for
  the selected instance. This is the reclass tool.

- **Merge selected** (button under the instance list, left side). Hold Ctrl and click
  two or more instances in the list so they are all selected, then press Merge. They
  become one instance. The largest one is kept, along with its attributes.

- **Lasso split** (button under the instance list). Select one instance first, then
  press Lasso split. Now left-click around the part you want to cut off, dropping a
  point at each click, and a green outline follows your cursor. When the outline is
  around the part, right-click (or double-click, or press Enter) to close it and cut.
  The points you enclosed light up yellow so you can check the selection, then you
  pick a class for the new piece. Press Esc to cancel without cutting.

- **Undo** (button next to Merge and Lasso, or press Ctrl+Z). Reverses your last
  reclass, split, or merge. You can press it more than once to step back.

## Which classes to focus on first

You will not have time to inspect everything with the same care, so put your effort
here first:

- **Blinds.** Check the class. Many blinds are really windows.
- **Windows.** Check the segmentation. This is where you find one instance that
  should be split, or a few that should be merged.
- **Walls.** Check for fragmentation. Walls often come in small broken pieces.

Do the other classes the same way, just at lower priority.

## After you fix an instance

When you change a class, the list of functional attributes that apply to that
instance changes too. So after a class change, look at the attribute checkboxes again
and make sure the right ones are on for the new class. Do not leave attributes ticked
that no longer make sense.

## Save vs Export (two different buttons)

There are two buttons and they do different things:

- **Save** (top right). Press this often as you work. It is quick and it writes your
  review and the corrected labels: the `.review.json`, the `.reviewed.h5` (this is the
  corrected file for the building), and if you made segmentation edits the
  `.edits.json` and `.edits.npz`. These are the files you upload.

- **Export corrected .laz** (right panel). This writes a full copy of the point cloud
  with your corrections baked in. It is a large file (over 100 MB) and takes a bit of
  time. You only need it if someone asks for the corrected cloud. It is not part of
  the normal upload, so do not run it every time.

When you Save, if any instance still has applicable attributes but none turned on, the
app warns you and lists those instance numbers. Go back and check them, or continue
anyway if they really have none.

## When you are done

Save your work, then upload to the AI server, in the same place we use for the reviews:

- your edit files: the `.review.json`, and the `.edits.json` / `.edits.npz`, and
- the corrected `.h5` file (`.reviewed.h5`) for the building.

That is the output of this round.

## Handy reminders

- Left-drag rotates, right-drag pans, wheel zooms.
- Ctrl+Z or the Undo button reverses your last reclass, split, or merge.
- Always confirm against the imagery or maps before you change a class or split.
- If the status bar shows "App update available", save your work, close the app, run
  `git pull`, then reopen so you have the latest version.
