import re


class NelBaseModifier:
    @staticmethod
    def extract_answer(text):
        text = text.replace(' - ', ' # ')
        pattern = r'([a-z0-9\s\-]+\s+[#]\s+(?:http://[a-z-0-9_\.\/]+;)*(?:http://[a-z-0-9_\.\/]+))'
        matches = re.findall(pattern=pattern, string=text, flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted entities: '
        if len(matches) >= 1:
            match_found = True
            for m in matches:
                match += f'{m.replace(' # ', ' - ')}, '
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class HansardNelModifier:
    @staticmethod
    def extract_answer(text):
        pattern = r'([a-z0-9\s]+\s*[:-]\s*(?:[a-z0-9\.]+\s\[[^\]]+\];\s?)*(?:[a-z0-9\.]+\s\[[^\]]+\]))'
        matches = re.findall(pattern=pattern, string=text, flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted entities: '
        if len(matches) >= 1:
            match_found = True
            for m in matches:
                m = re.sub(r'\s*\[[^]]+\]', '', m, 1000, re.IGNORECASE)
                match += f'{m.replace(': ', ' - ')}, '
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\*+', repl=' ', string=match, flags=re.IGNORECASE)
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class SaHansardNelModifier:
    @staticmethod
    def extract_answer(text):
        text = text.replace(' - ', ' # ')

        pattern = r'([a-z0-9\s\-()\.]+\s*[#]\s*(?:[a-z0-9\.];\s?)*(?:[a-z0-9\.]+))'

        matches = re.findall(pattern=pattern, string=text, flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted entities: '
        if len(matches) >= 1:
            match_found = True
            for m in matches:
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ')}, '
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\*+', repl=' ', string=match, flags=re.IGNORECASE)
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text

class FoodOnNelExtendedModifier:
    @staticmethod
    def extract_answer(text):
        text = text.replace(' - ', ' # ')
        pattern = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s*\**\s*(?:http[s]?://[a-z-0-9_\.\/#?:=]+;)*(?:http[s]?://[a-z-0-9_\.\/#?:=]+))'
        matches = re.findall(pattern=pattern, string=text, flags=re.IGNORECASE)

        pattern_2 = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s*(?:\[[a-z0-9:\s_\-#\/\.]+\]\(http[s]?://[a-z-0-9_\.\/:#?=]+\))*\[[a-z0-9:\s_\-#\/\.]+\]\(http[s]?://[a-z-0-9_\.\/:#?=]+\))'
        matches_2 = re.findall(pattern=pattern_2, string=text, flags=re.IGNORECASE)

        pattern_3 = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s*(?:[<`]?http://[a-z-0-9_\.\/#?:=]+[`>]?;)*(?:[`<]?http://[a-z-0-9_\.\/#?:=]+[`>]?))'
        matches_3 = re.findall(pattern=pattern_3, string=text, flags=re.IGNORECASE)

        pattern_4 = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s*`[a-z0-9:_\s]+`\s*.\s*[\[\(]?http[s]?://[a-z-0-9_\.\/:#?=]+[\]\)]?)'
        matches_4 = re.findall(pattern=pattern_4, string=text, flags=re.IGNORECASE)

        pattern_5 = r'(\*+[a-z0-9\s\-]+\*+\s*[:#]\s*[a-z0-9:_\s]+[\[\(]?http[s]?://[a-z-0-9_\.\/:#?=]+[\]\)]?)'
        matches_5 = re.findall(pattern=pattern_5, string=text, flags=re.IGNORECASE)

        pattern_6 = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s*`[a-z0-9:_\s]+`\s*.\s*[\[\(][a-z0-9\s]+[\]\)]\s*[\[\(]?http[s]?://[a-z-0-9_\.\/:#?=]+[\]\)]?)'
        matches_6 = re.findall(pattern=pattern_6, string=text, flags=re.IGNORECASE)

        pattern_7 = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s+[a-z0-9]+:[a-z0-9]+\s+(?:http[s]?://[a-z0-9_\.\/#?:=]+;)*(?:http[s]?://[a-z0-9_\.\/#?:=]+))'
        matches_7 = re.findall(pattern=pattern_7, string=text, flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted entities: '

        if len(matches_2) >= 1:
            match_found = True
            for m in matches_2:
                m = re.sub(pattern=r'\[[a-z0-9:\s_\-#\/\.]+\]', repl=' ', string=m, flags=re.IGNORECASE)
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '')}, '
        elif len(matches_3) >= 1:
            match_found = True
            for m in matches_3:
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace('`', '')}, '
        elif len(matches_4) >= 1:
            match_found = True
            for m in matches_4:
                m = re.sub(pattern=r'`[a-z0-9:_\s]+`\s*.\s*', repl=' ', string=m, flags=re.IGNORECASE)
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '').replace('`', '')}, '
        elif len(matches_5) >= 1:
            match_found = True
            for m in matches_5:
                m = re.sub(pattern=r'[:#]\s*[a-z0-9:_\s]+[\[\(]', repl=' ', string=m, flags=re.IGNORECASE)
                match += f'{m.replace('** ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '').replace('`', '')}, '
        elif len(matches_6) >= 1:
            match_found = True
            for m in matches_6:
                m = re.sub(pattern=r'`[a-z0-9:_\s]+`\s*.\s*[\[\(][a-z0-9\s]+[\]\)]', repl=' ', string=m, flags=re.IGNORECASE)
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '').replace('`', '')}, '
        elif len(matches_7) >= 1:
            match_found = True
            for m in matches_7:
                m = re.sub(pattern=r'\s+[a-z0-9]+:[a-z0-9]+\s+', repl=' ', string=m,
                           flags=re.IGNORECASE)
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '').replace(
                    '`', '')}, '
        elif len(matches) >= 1:
            match_found = True
            for m in matches:
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace('*', '')}, '
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\*+', repl=' ', string=match, flags=re.IGNORECASE)
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class SnomedNelExtendedModifier:
    @staticmethod
    def extract_answer(text):
        text = text.replace(' - ', ' # ')
        pattern = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s*\**\s*(?:http[s]?://[a-z-0-9_\.\/#?:=]+;)*(?:http[s]?://[a-z-0-9_\.\/#?:=]+))'
        matches = re.findall(pattern=pattern, string=text, flags=re.IGNORECASE)

        pattern_1 = r'([0-9]+\.\s+\**([a-z0-9\s\-]+)\**\s*[:#]\s*[^=]+[\[\(]?(http[s]?://[a-z-0-9_\.\/:#?=&\s]+)[\]\)]?)'
        matches_1 = re.findall(pattern=pattern_1, string=text, flags=re.IGNORECASE)

        pattern_2 = r'(\*+[a-z0-9\s\-]+\*+\s*[:#]\s*[a-z0-9:_\s]+[\[\(]?http[s]?://[a-z-0-9_\.\/:#?=]+[\]\)]?)'
        matches_2 = re.findall(pattern=pattern_2, string=text, flags=re.IGNORECASE)

        pattern_3 = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s+[\[\(]?[a-z0-9\s]+:\s*[a-z0-9]+[\]\)]?\s*(?:[\[\(]?http[s]?://[a-z-0-9_\.\/:#?=&]+[\]\)]?;)*(?:[\[\(]?http[s]?://[a-z-0-9_\.\/:#?=&]+[\]\)]?))'
        matches_3 = re.findall(pattern=pattern_3, string=text, flags=re.IGNORECASE)

        pattern_4 = r'(\**[a-z0-9\s\-]+\**\s*[:#]\s+[a-z0-9\s]+:?[a-z0-9]+\s+(?:http[s]?://[a-z0-9_\.\/#?:=]+;)*(?:http[s]?://[a-z0-9_\.\/#?:=]+))'
        matches_4 = re.findall(pattern=pattern_4, string=text, flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted entities: '

        if len(matches_2) >= 1:
            match_found = True
            for m in matches_2:
                m = re.sub(pattern=r'[:#]\s*[a-z0-9:_\s]+[\[\(]', repl=' ', string=m, flags=re.IGNORECASE)
                match += f'{m.replace('** ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '').replace('`', '')}, '
        elif len(matches_3) >= 1:
            match_found = True
            for m in matches_3:
                m = re.sub(pattern=r'[\[\(]?[a-z0-9\s]+:\s*[a-z0-9]+[\]\)]?', repl=' ', string=m, flags=re.IGNORECASE)
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '').replace('`', '')}, '
        elif len(matches_4) >= 1:
            match_found = True
            for m in matches_4:
                m = re.sub(pattern=r'\s+[a-z0-9]+:[a-z0-9]+\s+', repl=' ', string=m,
                           flags=re.IGNORECASE)
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace(
                    '(', '').replace(')', '').replace('*', '').replace('`', '')}, '
        elif len(matches) >= 1:
            match_found = True
            for m in matches:
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace('*', '')}, '
        elif len(matches_1) >= 1:
            match_found = True
            for m in matches_1:
                m = m[1] + ' - ' + m[2]
                match += f'{m.replace(': ', ' - ').replace(' # ', ' - ').replace('(', '').replace(
                    ')', '').replace('*', '').replace('`', '')}, '
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\*+', repl=' ', string=match, flags=re.IGNORECASE)
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class FsaBaseModifier:
    @staticmethod
    def extract_answer(text):
        pattern_fat = r'(fat\s+[-]\s+(?:red|green|orange))'
        pattern_salt = r'(salt\s+[-]\s+(?:red|green|orange))'
        pattern_saturates = r'(saturates\s+[-]\s+(?:red|green|orange))'
        pattern_sugars = r'(sugars\s+[-]\s+(?:red|green|orange))'

        matches_fat = re.findall(pattern=pattern_fat, string=text.lower(), flags=re.IGNORECASE)
        matches_salt = re.findall(pattern=pattern_salt, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates = re.findall(pattern=pattern_saturates, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars = re.findall(pattern=pattern_sugars, string=text.lower(), flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted FSA lights: '
        if len(matches_fat) >= 1:
            match_found = True
            match += f'{matches_fat[-1]}, '
        if len(matches_salt) >= 1:
            match_found = True
            match += f'{matches_salt[-1]}, '
        if len(matches_saturates) >= 1:
            match_found = True
            match += f'{matches_saturates[-1]}, '
        if len(matches_sugars) >= 1:
            match_found = True
            match += f'{matches_sugars[-1]}'
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class FsaExtendedModifier:
    @staticmethod
    def extract_answer(text):
        pattern_fat = r'(fat\s*[:-]\s*\**(?:red|green|orange))'
        pattern_salt = r'(salt\s*[:-]\s*\**(?:red|green|orange))'
        pattern_saturates = r'(saturates\s*[:-]\s*\**(?:red|green|orange))'
        pattern_sugars = r'(sugars\s*[:-]\s*\**(?:red|green|orange))'

        pattern_fat_2 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*fat[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_salt_2 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*salt[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_saturates_2 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*saturate[sd](?: fat)?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_sugars_2 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*sugar[s]?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'

        pattern_fat_3 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*fat[,\s][a-z0-9,\s\(\)\./%]*$)'
        pattern_salt_3 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*salt[,\s][a-z0-9,\s\(\)\./%]*$)'
        pattern_saturates_3 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*saturate[sd](?: fat)?[,\s][a-z0-9,\s\(\)\./%]*$)'
        pattern_sugars_3 = r'([\*+]\s+(red|green|orange):\s*[a-z0-9,\s\(\)\./%]+\s*sugar[s]?[,\s][a-z0-9,\s\(\)\./%]*$)'

        pattern_fat_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*fat[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_salt_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*salt[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_saturates_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*saturate[sd](?: fat)?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_sugars_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*sugar[s]?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'

        pattern_fat_5 = r'(fat[a-z,\s]+\((red|green|orange)\))'
        pattern_salt_5 = r'(salt[a-z,\s]+\((red|green|orange)\))'
        pattern_saturates_5 = r'(saturate[sd](?: fat)?[a-z,\s]+\((red|green|orange)\))'
        pattern_sugars_5 = r'(sugar[s]?[a-z,\s]+\((red|green|orange)\))'

        pattern_fat_6 = r'(recipe.*:.*fat\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange)\))'
        pattern_salt_6 = r'(recipe.*:.*salt\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange)\))'
        pattern_saturates_6 = r'(recipe.*:.*saturate[sd](?: fat)?\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange)\))'
        pattern_sugars_6 = r'(recipe.*:.*sugar[s]?\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange)\))'

        matches_fat = re.findall(pattern=pattern_fat, string=text.lower(), flags=re.IGNORECASE)
        matches_salt = re.findall(pattern=pattern_salt, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates = re.findall(pattern=pattern_saturates, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars = re.findall(pattern=pattern_sugars, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_2 = re.findall(pattern=pattern_fat_2, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_2 = re.findall(pattern=pattern_salt_2, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_2 = re.findall(pattern=pattern_saturates_2, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_2 = re.findall(pattern=pattern_sugars_2, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_3 = re.findall(pattern=pattern_fat_3, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_3 = re.findall(pattern=pattern_salt_3, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_3 = re.findall(pattern=pattern_saturates_3, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_3 = re.findall(pattern=pattern_sugars_3, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_4 = re.findall(pattern=pattern_fat_4, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_4 = re.findall(pattern=pattern_salt_4, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_4 = re.findall(pattern=pattern_saturates_4, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_4 = re.findall(pattern=pattern_sugars_4, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_5 = re.findall(pattern=pattern_fat_5, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_5 = re.findall(pattern=pattern_salt_5, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_5 = re.findall(pattern=pattern_saturates_5, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_5 = re.findall(pattern=pattern_sugars_5, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_6 = re.findall(pattern=pattern_fat_6, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_6 = re.findall(pattern=pattern_salt_6, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_6 = re.findall(pattern=pattern_saturates_6, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_6 = re.findall(pattern=pattern_sugars_6, string=text.lower(), flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted FSA lights: '
        if len(matches_fat) >= 1:
            match_found = True
            match += f'{matches_fat[-1].replace(':', ' - ')}, '
        elif len(matches_fat_2) >= 1:
            match_found = True
            match += f'fat - {matches_fat_2[-1][-1]}, '
        elif len(matches_fat_3) >= 1:
            match_found = True
            match += f'fat - {matches_fat_3[-1][-1]}, '
        elif len(matches_fat_4) >= 1:
            match_found = True
            match += f'fat - {matches_fat_4[-1][-1]}, '
        elif len(matches_fat_5) >= 1:
            match_found = True
            match += f'fat - {matches_fat_5[-1][-1]}, '
        elif len(matches_fat_6) >= 1:
            match_found = True
            match += f'fat - {matches_fat_6[-1][-1]}, '

        if len(matches_salt) >= 1:
            match_found = True
            match += f'{matches_salt[-1].replace(':', ' - ')}, '
        elif len(matches_salt_2) >= 1:
            match_found = True
            match += f'salt - {matches_salt_2[-1][-1]}, '
        elif len(matches_salt_3) >= 1:
            match_found = True
            match += f'salt - {matches_salt_3[-1][-1]}, '
        elif len(matches_salt_4) >= 1:
            match_found = True
            match += f'salt - {matches_salt_4[-1][-1]}, '
        elif len(matches_salt_5) >= 1:
            match_found = True
            match += f'salt - {matches_salt_5[-1][-1]}, '
        elif len(matches_salt_6) >= 1:
            match_found = True
            match += f'salt - {matches_salt_6[-1][-1]}, '

        if len(matches_saturates) >= 1:
            match_found = True
            match += f'{matches_saturates[-1].replace(':', ' - ')}, '
        elif len(matches_saturates_2) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_2[-1][-1]}, '
        elif len(matches_saturates_3) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_3[-1][-1]}, '
        elif len(matches_saturates_4) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_4[-1][-1]}, '
        elif len(matches_saturates_5) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_5[-1][-1]}, '
        elif len(matches_saturates_6) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_6[-1][-1]}, '

        if len(matches_sugars) >= 1:
            match_found = True
            match += f'{matches_sugars[-1].replace(':', ' - ')}'
        elif len(matches_sugars_2) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_2[-1][-1]}, '
        elif len(matches_sugars_3) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_3[-1][-1]}, '
        elif len(matches_sugars_4) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_4[-1][-1]}, '
        elif len(matches_sugars_5) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_5[-1][-1]}, '
        elif len(matches_sugars_6) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_6[-1][-1]}, '

        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\*+', repl=' ', string=match, flags=re.IGNORECASE)
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class FsaSynonymModifier:
    @staticmethod
    def extract_answer(text):
        pattern_fat = r'(fat\s*[:-]\s*\**(?:red|green|orange|amber|yellow|low|medium|moderate|high))'
        pattern_salt = r'(salt\s*[:-]\s*\**(?:red|green|orange|amber|yellow|low|medium|moderate|high))'
        pattern_saturates = r'(saturates\s*[:-]\s*\**(?:red|green|orange|amber|yellow|low|medium|moderate|high))'
        pattern_sugars = r'(sugars\s*[:-]\s*\**(?:red|green|orange|amber|yellow|low|medium|moderate|high))'

        pattern_fat_2 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*fat[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_salt_2 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*salt[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_saturates_2 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*saturate[sd](?: fat)?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_sugars_2 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*sugar[s]?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'

        pattern_fat_3 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*fat[,\s][a-z0-9,\s\(\)\./%]*$)'
        pattern_salt_3 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*salt[,\s][a-z0-9,\s\(\)\./%]*$)'
        pattern_saturates_3 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*saturate[sd](?: fat)?[,\s][a-z0-9,\s\(\)\./%]*$)'
        pattern_sugars_3 = r'([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high):\s*[a-z0-9,\s\(\)\./%]+\s*sugar[s]?[,\s][a-z0-9,\s\(\)\./%]*$)'

        pattern_fat_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*fat[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_salt_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*salt[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_saturates_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*saturate[sd](?: fat)?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'
        pattern_sugars_4 = r'recipe:[a-z,\s\(\)]+([\*+]\s+(red|green|orange|amber|yellow|low|medium|moderate|high)(?:\s+\([a-z]+\))?\s+for\s*[a-z0-9,\s\(\)\./%]+\s*sugar[s]?[,\s][a-z0-9,\s\(\)\./%]*[\*\s])'

        pattern_fat_5 = r'(fat[a-z,\s]+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'
        pattern_salt_5 = r'(salt[a-z,\s]+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'
        pattern_saturates_5 = r'(saturate[sd](?: fat)?[a-z,\s]+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'
        pattern_sugars_5 = r'(sugar[s]?[a-z,\s]+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'

        pattern_fat_6 = r'(recipe.*:.*fat\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'
        pattern_salt_6 = r'(recipe.*:.*salt\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'
        pattern_saturates_6 = r'(recipe.*:.*saturate[sd](?: fat)?\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'
        pattern_sugars_6 = r'(recipe.*:.*sugar[s]?\**:?\**\s+[0-9\.%gper/]+g?\s+\((red|green|orange|amber|yellow|low|medium|moderate|high)\))'

        matches_fat = re.findall(pattern=pattern_fat, string=text.lower(), flags=re.IGNORECASE)
        matches_salt = re.findall(pattern=pattern_salt, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates = re.findall(pattern=pattern_saturates, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars = re.findall(pattern=pattern_sugars, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_2 = re.findall(pattern=pattern_fat_2, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_2 = re.findall(pattern=pattern_salt_2, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_2 = re.findall(pattern=pattern_saturates_2, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_2 = re.findall(pattern=pattern_sugars_2, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_3 = re.findall(pattern=pattern_fat_3, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_3 = re.findall(pattern=pattern_salt_3, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_3 = re.findall(pattern=pattern_saturates_3, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_3 = re.findall(pattern=pattern_sugars_3, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_4 = re.findall(pattern=pattern_fat_4, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_4 = re.findall(pattern=pattern_salt_4, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_4 = re.findall(pattern=pattern_saturates_4, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_4 = re.findall(pattern=pattern_sugars_4, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_5 = re.findall(pattern=pattern_fat_5, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_5 = re.findall(pattern=pattern_salt_5, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_5 = re.findall(pattern=pattern_saturates_5, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_5 = re.findall(pattern=pattern_sugars_5, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_6 = re.findall(pattern=pattern_fat_6, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_6 = re.findall(pattern=pattern_salt_6, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_6 = re.findall(pattern=pattern_saturates_6, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_6 = re.findall(pattern=pattern_sugars_6, string=text.lower(), flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted FSA lights: '
        if len(matches_fat) >= 1:
            match_found = True
            match += f'{matches_fat[-1].replace(':', ' - ')}, '
        elif len(matches_fat_2) >= 1:
            match_found = True
            match += f'fat - {matches_fat_2[-1][-1]}, '
        elif len(matches_fat_3) >= 1:
            match_found = True
            match += f'fat - {matches_fat_3[-1][-1]}, '
        elif len(matches_fat_4) >= 1:
            match_found = True
            match += f'fat - {matches_fat_4[-1][-1]}, '
        elif len(matches_fat_5) >= 1:
            match_found = True
            match += f'fat - {matches_fat_5[-1][-1]}, '
        elif len(matches_fat_6) >= 1:
            match_found = True
            match += f'fat - {matches_fat_6[-1][-1]}, '

        if len(matches_salt) >= 1:
            match_found = True
            match += f'{matches_salt[-1].replace(':', ' - ')}, '
        elif len(matches_salt_2) >= 1:
            match_found = True
            match += f'salt - {matches_salt_2[-1][-1]}, '
        elif len(matches_salt_3) >= 1:
            match_found = True
            match += f'salt - {matches_salt_3[-1][-1]}, '
        elif len(matches_salt_4) >= 1:
            match_found = True
            match += f'salt - {matches_salt_4[-1][-1]}, '
        elif len(matches_salt_5) >= 1:
            match_found = True
            match += f'salt - {matches_salt_5[-1][-1]}, '
        elif len(matches_salt_6) >= 1:
            match_found = True
            match += f'salt - {matches_salt_6[-1][-1]}, '

        if len(matches_saturates) >= 1:
            match_found = True
            match += f'{matches_saturates[-1].replace(':', ' - ')}, '
        elif len(matches_saturates_2) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_2[-1][-1]}, '
        elif len(matches_saturates_3) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_3[-1][-1]}, '
        elif len(matches_saturates_4) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_4[-1][-1]}, '
        elif len(matches_saturates_5) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_5[-1][-1]}, '
        elif len(matches_saturates_6) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_6[-1][-1]}, '

        if len(matches_sugars) >= 1:
            match_found = True
            match += f'{matches_sugars[-1].replace(':', ' - ')}'
        elif len(matches_sugars_2) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_2[-1][-1]}, '
        elif len(matches_sugars_3) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_3[-1][-1]}, '
        elif len(matches_sugars_4) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_4[-1][-1]}, '
        elif len(matches_sugars_5) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_5[-1][-1]}, '
        elif len(matches_sugars_6) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_6[-1][-1]}, '

        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\*+', repl=' ', string=match, flags=re.IGNORECASE)
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
            match = match.replace('amber', 'orange')
            match = match.replace('yellow', 'orange')
            match = match.replace('medium', 'orange')
            match = match.replace('moderate', 'orange')

            match = match.replace('low', 'green')
            match = match.replace('high', 'red')
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class NutrientBaseModifier:
    @staticmethod
    def extract_answer(text):
        pattern_fat = r'(fat\s+[-]\s+[0-9]+\.[0-9]+)'
        pattern_protein = r'(protein\s+[-]\s+[0-9]+\.[0-9]+)'
        pattern_salt = r'(salt\s+[-]\s+[0-9]+\.[0-9]+)'
        pattern_saturates = r'(saturates\s+[-]\s+[0-9]+\.[0-9]+)'
        pattern_sugars = r'(sugars\s+[-]\s+[0-9]+\.[0-9]+)'

        matches_fat = re.findall(pattern=pattern_fat, string=text.lower(), flags=re.IGNORECASE)
        matches_protein = re.findall(pattern=pattern_protein, string=text.lower(), flags=re.IGNORECASE)
        matches_salt = re.findall(pattern=pattern_salt, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates = re.findall(pattern=pattern_saturates, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars = re.findall(pattern=pattern_sugars, string=text.lower(), flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted nutrient values: '
        if len(matches_fat) >= 1:
            match_found = True
            match += f'{matches_fat[-1]}, '
        if len(matches_protein) >= 1:
            match_found = True
            match += f'{matches_protein[-1]}, '
        if len(matches_salt) >= 1:
            match_found = True
            match += f'{matches_salt[-1]}, '
        if len(matches_saturates) >= 1:
            match_found = True
            match += f'{matches_saturates[-1]}, '
        if len(matches_sugars) >= 1:
            match_found = True
            match += f'{matches_sugars[-1]}'
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text


class NutrientExtendedModifier:
    @staticmethod
    def extract_answer(text):
        pattern_fat = r'(fat\s*[:-]\s*\**([0-9]+\.[0-9]+))'
        pattern_protein = r'(protein\s*[:-]\s*\**([0-9]+\.[0-9]+))'
        pattern_salt = r'(salt\s*[:-]\s*\**([0-9]+\.[0-9]+))'
        pattern_saturates = r'(saturates\s*[:-]\s*\**([0-9]+\.[0-9]+))'
        pattern_sugars = r'(sugar[s]?\s*[:-]\s*\**([0-9]+\.[0-9]+))'

        pattern_fat_2 = r'(fat:?\s+(?:[a-z\s]+)?[\(]?(?:approximately|estimated)[\)]?\s+([0-9]+\.[0-9]+)(?:g.100g)?)'
        pattern_protein_2 = r'(protein:?\s+(?:[a-z\s]+)?[\(]?(?:approximately|estimated)[\)]?\s+([0-9]+\.[0-9]+)(?:g.100g)?)'
        pattern_salt_2 = r'(salt:?\s+(?:[a-z\s]+)?[\(]?(?:approximately|estimated)[\)]?\s+([0-9]+\.[0-9]+)(?:g.100g)?)'
        pattern_saturates_2 = r'(saturates:?\s+(?:[a-z\s]+)?[\(]?(?:approximately|estimated)[\)]?\s+([0-9]+\.[0-9]+)(?:g.100g)?)'
        pattern_sugars_2 = r'(sugar[s]?:?\s+(?:[a-z\s]+)?[\(]?(?:approximately|estimated)[\)]?\s+([0-9]+\.[0-9]+)(?:g.100g)?)'

        pattern_fat_3 = r'(Total fat content: [0-9\.]+g Per 100g: ([0-9]+\.[0-9]+)g)'
        pattern_protein_3 = r'(Total protein content: [0-9\.]+g Per 100g: ([0-9]+\.[0-9]+)g)'
        pattern_salt_3 = r'(Total salt content: [0-9\.]+g Per 100g: ([0-9]+\.[0-9]+)g)'
        pattern_saturates_3 = r'(Total saturated fat content: [0-9\.]+g Per 100g: ([0-9]+\.[0-9]+)g)'
        pattern_sugars_3 = r'(Total sugar content: [0-9\.]+g Per 100g: ([0-9]+\.[0-9]+)g)'

        pattern_fat_4 = r'(Fat:?(?: per 100 grams is)? approximately ([0-9]+\.[0-9]+))'
        pattern_protein_4 = r'(Protein:?(?: per 100 grams is)? approximately ([0-9]+\.[0-9]+))'
        pattern_salt_4 = r'(Salt:?(?: per 100 grams is)? approximately ([0-9]+\.[0-9]+))'
        pattern_saturates_4 = r'(Saturates:?(?: per 100 grams is)? approximately ([0-9]+\.[0-9]+))'
        pattern_sugars_4 = r'(Sugar[s]?:?(?: per 100 grams is)? approximately ([0-9]+\.[0-9]+))'

        pattern_fat_5 = r'(Fat:\s+(?:[a-z0-9\[\]\s\(\)\.\/+,%]+)?\s*=\s*([0-9]+\.[0-9]+))'
        pattern_protein_5 = r'(Protein:\s+(?:[a-z0-9\[\]\s\(\)\.\/+,%]+)?\s*=\s*([0-9]+\.[0-9]+))'
        pattern_salt_5 = r'(Salt:\s+(?:[a-z0-9\[\]\s\(\)\.\/+,%]+)?\s*=\s*([0-9]+\.[0-9]+))'
        pattern_saturates_5 = r'(Saturates:\s+(?:[a-z0-9\[\]\s\(\)\.\/+,%]+)?\s*=\s*([0-9]+\.[0-9]+))'
        pattern_sugars_5 = r'(Sugar[s]?:\s+(?:[a-z0-9\[\]\s\(\)\.\/+,%]+)?\s*=\s*([0-9]+\.[0-9]+))'

        pattern_fat_6 = r'(Fat:\s+[\(](?:from [a-z\s]+)[\)]\s+([0-9]+\.[0-9]+) g\s+[\(](?:[a-z0-9\s\.,%]+)[\)])'
        pattern_protein_6 = r'(Protein:\s+[\(](?:from [a-z\s]+)[\)]\s+([0-9]+\.[0-9]+) g\s+[\(](?:[a-z0-9\s\.,%]+)[\)])'
        pattern_salt_6 = r'(Salt:\s+[\(](?:from [a-z\s]+)[\)]\s+([0-9]+\.[0-9]+) g\s+[\(](?:[a-z0-9\s\.,%]+)[\)])'
        pattern_saturates_6 = r'(Saturates:\s+[\(](?:from [a-z\s]+)[\)]\s+([0-9]+\.[0-9]+) g\s+[\(](?:[a-z0-9\s\.,%]+)[\)])'
        pattern_sugars_6 = r'(Sugar[s]?:\s+[\(](?:from [a-z\s]+)[\)]\s+([0-9]+\.[0-9]+) g\s+[\(](?:[a-z0-9\s\.,%]+)[\)])'

        matches_fat = re.findall(pattern=pattern_fat, string=text.lower(), flags=re.IGNORECASE)
        matches_protein = re.findall(pattern=pattern_protein, string=text.lower(), flags=re.IGNORECASE)
        matches_salt = re.findall(pattern=pattern_salt, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates = re.findall(pattern=pattern_saturates, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars = re.findall(pattern=pattern_sugars, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_2 = re.findall(pattern=pattern_fat_2, string=text.lower(), flags=re.IGNORECASE)
        matches_protein_2 = re.findall(pattern=pattern_protein_2, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_2 = re.findall(pattern=pattern_salt_2, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_2 = re.findall(pattern=pattern_saturates_2, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_2 = re.findall(pattern=pattern_sugars_2, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_3 = re.findall(pattern=pattern_fat_3, string=text.lower(), flags=re.IGNORECASE)
        matches_protein_3 = re.findall(pattern=pattern_protein_3, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_3 = re.findall(pattern=pattern_salt_3, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_3 = re.findall(pattern=pattern_saturates_3, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_3 = re.findall(pattern=pattern_sugars_3, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_4 = re.findall(pattern=pattern_fat_4, string=text.lower(), flags=re.IGNORECASE)
        matches_protein_4 = re.findall(pattern=pattern_protein_4, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_4 = re.findall(pattern=pattern_salt_4, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_4 = re.findall(pattern=pattern_saturates_4, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_4 = re.findall(pattern=pattern_sugars_4, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_5 = re.findall(pattern=pattern_fat_5, string=text.lower(), flags=re.IGNORECASE)
        matches_protein_5 = re.findall(pattern=pattern_protein_5, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_5 = re.findall(pattern=pattern_salt_5, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_5 = re.findall(pattern=pattern_saturates_5, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_5 = re.findall(pattern=pattern_sugars_5, string=text.lower(), flags=re.IGNORECASE)

        matches_fat_6 = re.findall(pattern=pattern_fat_6, string=text.lower(), flags=re.IGNORECASE)
        matches_protein_6 = re.findall(pattern=pattern_protein_6, string=text.lower(), flags=re.IGNORECASE)
        matches_salt_6 = re.findall(pattern=pattern_salt_6, string=text.lower(), flags=re.IGNORECASE)
        matches_saturates_6 = re.findall(pattern=pattern_saturates_6, string=text.lower(), flags=re.IGNORECASE)
        matches_sugars_6 = re.findall(pattern=pattern_sugars_6, string=text.lower(), flags=re.IGNORECASE)

        match_found = False
        match = 'Predicted nutrient values: '
        if len(matches_fat) >= 1:
            match_found = True
            match += f'fat - {matches_fat[0][-1]}, '
        elif len(matches_fat_2) >= 1:
            match_found = True
            match += f'fat - {matches_fat_2[0][-1]}, '
        elif len(matches_fat_3) >= 1:
            match_found = True
            match += f'fat - {matches_fat_3[0][-1]}, '
        elif len(matches_fat_4) >= 1:
            match_found = True
            match += f'fat - {matches_fat_4[0][-1]}, '
        elif len(matches_fat_5) >= 1:
            match_found = True
            match += f'fat - {matches_fat_5[0][-1]}, '
        elif len(matches_fat_6) >= 1:
            match_found = True
            match += f'fat - {matches_fat_6[0][-1]}, '
        if len(matches_protein) >= 1:
            match_found = True
            match += f'protein - {matches_protein[0][-1]}, '
        elif len(matches_protein_2) >= 1:
            match_found = True
            match += f'protein - {matches_protein_2[0][-1]}, '
        elif len(matches_protein_3) >= 1:
            match_found = True
            match += f'protein - {matches_protein_3[0][-1]}, '
        elif len(matches_protein_4) >= 1:
            match_found = True
            match += f'protein - {matches_protein_4[0][-1]}, '
        elif len(matches_protein_5) >= 1:
            match_found = True
            match += f'protein - {matches_protein_5[0][-1]}, '
        elif len(matches_protein_6) >= 1:
            match_found = True
            match += f'protein - {matches_protein_6[0][-1]}, '
        if len(matches_salt) >= 1:
            match_found = True
            match += f'salt - {matches_salt[0][-1]}, '
        elif len(matches_salt_2) >= 1:
            match_found = True
            match += f'salt - {matches_salt_2[0][-1]}, '
        elif len(matches_salt_3) >= 1:
            match_found = True
            match += f'salt - {matches_salt_3[0][-1]}, '
        elif len(matches_salt_4) >= 1:
            match_found = True
            match += f'salt - {matches_salt_4[0][-1]}, '
        elif len(matches_salt_5) >= 1:
            match_found = True
            match += f'salt - {matches_salt_5[0][-1]}, '
        elif len(matches_salt_6) >= 1:
            match_found = True
            match += f'salt - {matches_salt_6[0][-1]}, '
        if len(matches_saturates) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates[0][-1]}, '
        elif len(matches_saturates_2) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_2[0][-1]}, '
        elif len(matches_saturates_3) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_3[0][-1]}, '
        elif len(matches_saturates_4) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_4[0][-1]}, '
        elif len(matches_saturates_5) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_5[0][-1]}, '
        elif len(matches_saturates_6) >= 1:
            match_found = True
            match += f'saturates - {matches_saturates_6[0][-1]}, '
        if len(matches_sugars) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars[0][-1]}'
        elif len(matches_sugars_2) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_2[0][-1]}'
        elif len(matches_sugars_3) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_3[0][-1]}'
        elif len(matches_sugars_4) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_4[0][-1]}'
        elif len(matches_sugars_5) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_5[0][-1]}'
        elif len(matches_sugars_6) >= 1:
            match_found = True
            match += f'sugars - {matches_sugars_6[0][-1]}'
        if match_found and match.endswith(', '):
            match = match[:-2]
        if match_found:
            match = re.sub(pattern=r'\*+', repl=' ', string=match, flags=re.IGNORECASE)
            match = re.sub(pattern=r'\s+', repl=' ', string=match, flags=re.IGNORECASE)
        if not match_found:
            match = ''
        return match

    def clean(self, text):
        text = self.extract_answer(text=text)
        return text
