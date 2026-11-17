def has_extractable_pdf_text(page, text: str) -> bool:
    if sum(char.isalnum() for char in text) < 20:
        return False
    if not page.get_images(full=True):
        return True

    text_blocks = [
        block for block in page.get_text("blocks", sort=True)
        if len(block) > 4
        and isinstance(block[4], str)
        and any(char.isalnum() for char in block[4])
    ]
    if not text_blocks or page.rect.is_empty:
        return False
    left = min(block[0] for block in text_blocks)
    top = min(block[1] for block in text_blocks)
    right = max(block[2] for block in text_blocks)
    bottom = max(block[3] for block in text_blocks)
    text_coverage = (
        max(0.0, right - left) * max(0.0, bottom - top)
        / (page.rect.width * page.rect.height)
    )
    return text_coverage >= 0.02
