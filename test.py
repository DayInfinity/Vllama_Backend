from functions.translate.translate import translate_fast

if __name__ == "__main__":
    text = """
    People experience many things every day, but they do not always notice them. When someone walks outside, they may see buildings, trees, or cars, but they often do not think about these things. They are used to them, so they feel normal. Even small changes, like a cool breeze or a warm sun, can affect how a person feels, but these changes are easy to ignore.

If a person slows down and pays attention, they can see more details. Colors look brighter, and sounds become clearer. A simple walk can feel more interesting when someone focuses on what is around them. They may hear birds, notice the shape of leaves, or feel the ground under their feet. These small observations can make everyday life more enjoyable.

People often move quickly because they are busy. They think about work, school, or other tasks. But taking a moment to observe the world can help reduce stress. It can make a person feel calm and relaxed. Paying attention to simple things is an easy way to feel more connected to the present moment.

Learning to notice everyday details is a skill. With practice, anyone can become more aware of their surroundings and enjoy daily life more fully.
"""

    translated = translate_fast(
        text=text,
        input_lang="en",
        output_lang="es",
    )

    print(translated)
