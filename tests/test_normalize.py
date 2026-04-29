from burnin_subtitle_checker.normalize import normalize_text


def test_normalize_preserves_indic_scripts_and_marks():
    assert normalize_text("  वो   कहाँ गई थी?! ") == "वो कहाँ गई थी"
    assert normalize_text("ಅವಳು, ಎಲ್ಲಿಗೆ? ಹೋದಳು!") == "ಅವಳು ಎಲ್ಲಿಗೆ ಹೋದಳು"


def test_normalize_removes_zero_width_joiners():
    assert normalize_text("वो\u200d कहाँ") == "वो कहाँ"
